"""TimeGAN en PyTorch.

Port del timegan.py oficial (TensorFlow 1.x) de:
Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks", NeurIPS 2019.
https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Mantiene la misma estructura: 5 redes (embedder, recovery, generator,
supervisor, discriminator), las mismas pérdidas y las 3 fases de entrenamiento.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# ---------------------------------------------------------------------------
# Utilidades (equivalentes a utils.py del repo oficial)
# ---------------------------------------------------------------------------
def extract_time(data):
    """Longitud de cada secuencia y longitud máxima."""
    time = [len(seq) for seq in data]
    return time, max(time)


def random_generator(batch_size, z_dim, T_mb, max_seq_len):
    """Ruido U(0,1) por paso de tiempo; ceros en el padding."""
    Z = np.zeros((batch_size, max_seq_len, z_dim), dtype=np.float32)
    for i in range(batch_size):
        Z[i, :T_mb[i], :] = np.random.uniform(0.0, 1.0, (T_mb[i], z_dim))
    return Z


def batch_generator(data, time, batch_size):
    """Mini-batch aleatorio."""
    idx = np.random.permutation(len(data))[:batch_size]
    X_mb = np.stack([data[i] for i in idx]).astype(np.float32)
    T_mb = [time[i] for i in idx]
    return X_mb, T_mb


def MinMaxScaler(data):
    """Normalización min-max por feature (igual que el original).

    Se ajusta con los datos que recibe ``timegan``: pásale solo el split de
    entrenamiento (``fit``) para que stop/tune/test no influyan en la escala.
    """
    if not np.isfinite(data).all():
        raise ValueError("ori_data tiene NaN/inf: revisa el tensor antes de entrenar.")
    min_val = np.min(np.min(data, axis=0), axis=0)
    data = data - min_val
    max_val = np.max(np.max(data, axis=0), axis=0)
    norm_data = data / (max_val + 1e-7)
    return norm_data, min_val, max_val


# ---------------------------------------------------------------------------
# Bloque base: RNN multicapa + capa densa
# (equivale a MultiRNNCell + dynamic_rnn + fully_connected)
# ---------------------------------------------------------------------------
class RNNNet(nn.Module):
    def __init__(self, module_name, input_dim, hidden_dim, num_layers,
                 output_dim, activation):
        super().__init__()
        rnn_cls = {"gru": nn.GRU, "lstm": nn.LSTM}[module_name]
        self.rnn = rnn_cls(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x, T):
        # pack/pad reproduce el sequence_length de dynamic_rnn:
        # la RNN no procesa el padding y sus salidas ahí quedan en cero.
        max_len = x.size(1)
        packed = pack_padded_sequence(x, T.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=max_len)
        out = self.fc(out)
        return self.activation(out) if self.activation is not None else out


class TimeGAN(nn.Module):
    def __init__(self, dim, hidden_dim, num_layers, module_name="gru"):
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers debe ser >= 2 (el supervisor usa num_layers-1).")
        z_dim = dim
        sig = torch.sigmoid
        # X -> H
        self.embedder = RNNNet(module_name, dim, hidden_dim, num_layers, hidden_dim, sig)
        # H -> X_tilde
        self.recovery = RNNNet(module_name, hidden_dim, hidden_dim, num_layers, dim, sig)
        # Z -> E_hat (latente sintético)
        self.generator = RNNNet(module_name, z_dim, hidden_dim, num_layers, hidden_dim, sig)
        # H_{<=t} -> H_{t+1}
        self.supervisor = RNNNet(module_name, hidden_dim, hidden_dim, num_layers - 1, hidden_dim, sig)
        # H -> logit por paso de tiempo (sin activación)
        self.discriminator = RNNNet(module_name, hidden_dim, hidden_dim, num_layers, 1, None)


# ---------------------------------------------------------------------------
# Función principal (misma firma que el original)
# ---------------------------------------------------------------------------
def timegan(ori_data, parameters):
    """Entrena TimeGAN y devuelve datos sintéticos.

    Args:
      ori_data: array (no, seq_len, dim) con las series originales.
      parameters: dict con hidden_dim, num_layer, iterations, batch_size,
                  module ('gru' o 'lstm') y opcionalmente device, seed y
                  n_samples (cuántas series generar; por defecto ``no``).

    Returns:
      generated_data: lista de arrays (seq_len_i, dim) en la escala original.
      La salida es continua: para tensores de w2 aplica
      ``w2_sequences.postprocess`` (flags 0/1, máscaras por canal).
    """
    device = torch.device(parameters.get(
        "device", "cuda" if torch.cuda.is_available() else "cpu"))
    seed = parameters.get("seed")
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    ori_data = np.asarray(ori_data, dtype=np.float32)
    no, seq_len, dim = ori_data.shape
    ori_time, max_seq_len = extract_time(ori_data)
    ori_data, min_val, max_val = MinMaxScaler(ori_data)

    hidden_dim = parameters["hidden_dim"]
    num_layers = parameters["num_layer"]
    iterations = parameters["iterations"]
    batch_size = parameters["batch_size"]
    module_name = parameters["module"]
    z_dim = dim
    gamma = 1

    model = TimeGAN(dim, hidden_dim, num_layers, module_name).to(device)
    emb, rec, gen, sup, disc = (model.embedder, model.recovery, model.generator,
                                model.supervisor, model.discriminator)

    er_params = list(emb.parameters()) + list(rec.parameters())
    gs_params = list(gen.parameters()) + list(sup.parameters())

    # Un optimizador por "solver", igual que en TF
    E0_solver = torch.optim.Adam(er_params)
    E_solver = torch.optim.Adam(er_params)
    D_solver = torch.optim.Adam(disc.parameters())
    G_solver = torch.optim.Adam(gs_params)
    GS_solver = torch.optim.Adam(gs_params)

    # --- helpers ---------------------------------------------------------
    def sample_batch(with_z=False):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X = torch.from_numpy(X_mb).to(device)
        T = torch.tensor(T_mb, dtype=torch.long)
        if not with_z:
            return X, T
        Z = torch.from_numpy(random_generator(len(T_mb), z_dim, T_mb, max_seq_len)).to(device)
        return X, T, Z

    def bce(logits, target_value):
        return F.binary_cross_entropy_with_logits(
            logits, torch.full_like(logits, target_value))

    def supervised_loss(H, T):
        # Predecir H_{t+1} a partir de H_{<=t}
        H_hat_supervise = sup(H, T)
        return F.mse_loss(H_hat_supervise[:, :-1, :], H[:, 1:, :])

    def step(loss, optimizer):
        # zero_grad sobre TODO el modelo: la pérdida puede propagar gradiente
        # por redes que este optimizador no actualiza (como var_list en TF).
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # --- 1. Entrenamiento embedder + recovery (autoencoder) ----------------
    print("Start Embedding Network Training")
    for itt in range(iterations):
        X, T = sample_batch()
        X_tilde = rec(emb(X, T), T)
        E_loss_T0 = F.mse_loss(X_tilde, X)
        step(10 * torch.sqrt(E_loss_T0), E0_solver)
        if itt % 1000 == 0:
            print(f"step: {itt}/{iterations}, e_loss: {np.sqrt(E_loss_T0.item()):.4f}")
    print("Finish Embedding Network Training")

    # --- 2. Solo pérdida supervisada (supervisor) --------------------------
    print("Start Training with Supervised Loss Only")
    for itt in range(iterations):
        X, T = sample_batch()
        with torch.no_grad():
            H = emb(X, T)  # en TF el embedder no se actualiza en esta fase
        G_loss_S = supervised_loss(H, T)
        step(G_loss_S, GS_solver)
        if itt % 1000 == 0:
            print(f"step: {itt}/{iterations}, s_loss: {np.sqrt(G_loss_S.item()):.4f}")
    print("Finish Training with Supervised Loss Only")

    # --- 3. Entrenamiento conjunto -----------------------------------------
    print("Start Joint Training")
    step_d_loss = float("nan")
    for itt in range(iterations):
        # Generador dos veces por cada paso del discriminador
        for _ in range(2):
            X, T, Z = sample_batch(with_z=True)

            # Generator + supervisor
            H = emb(X, T).detach()
            E_hat = gen(Z, T)
            H_hat = sup(E_hat, T)
            X_hat = rec(H_hat, T)
            Y_fake = disc(H_hat, T)
            Y_fake_e = disc(E_hat, T)

            G_loss_U = bce(Y_fake, 1.0)
            G_loss_U_e = bce(Y_fake_e, 1.0)
            G_loss_S = supervised_loss(H, T)

            # Momentos sobre el eje del batch (tf.nn.moments(x, [0]))
            G_loss_V1 = torch.mean(torch.abs(
                torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6)
                - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
            G_loss_V2 = torch.mean(torch.abs(X_hat.mean(dim=0) - X.mean(dim=0)))
            G_loss_V = G_loss_V1 + G_loss_V2

            G_loss = (G_loss_U + gamma * G_loss_U_e
                      + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V)
            step(G_loss, G_solver)

            # Embedder + recovery
            H = emb(X, T)
            X_tilde = rec(H, T)
            E_loss_T0 = F.mse_loss(X_tilde, X)
            E_loss = 10 * torch.sqrt(E_loss_T0) + 0.1 * supervised_loss(H, T)
            step(E_loss, E_solver)

        # Discriminador
        X, T, Z = sample_batch(with_z=True)
        with torch.no_grad():
            H = emb(X, T)
            E_hat = gen(Z, T)
            H_hat = sup(E_hat, T)
        D_loss = (bce(disc(H, T), 1.0)
                  + bce(disc(H_hat, T), 0.0)
                  + gamma * bce(disc(E_hat, T), 0.0))
        # Solo se entrena si el discriminador "no está funcionando bien"
        if D_loss.item() > 0.15:
            step(D_loss, D_solver)
            step_d_loss = D_loss.item()

        if itt % 1000 == 0:
            print(f"step: {itt}/{iterations}, d_loss: {step_d_loss:.4f}, "
                  f"g_loss_u: {G_loss_U.item():.4f}, "
                  f"g_loss_s: {np.sqrt(G_loss_S.item()):.4f}, "
                  f"g_loss_v: {G_loss_V.item():.4f}, "
                  f"e_loss_t0: {np.sqrt(E_loss_T0.item()):.4f}")
    print("Finish Joint Training")

    # --- Generación de datos sintéticos ------------------------------------
    generated_data = []
    n_samples = parameters.get("n_samples", no)
    # Longitudes de las series sintéticas: muestreadas de las originales
    # (en w2 todas miden 90 días).
    gen_time = [ori_time[i] for i in np.random.randint(0, no, n_samples)]
    model.eval()
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):  # por bloques para no saturar memoria
            t_chunk = gen_time[start:start + batch_size]
            Z = torch.from_numpy(
                random_generator(len(t_chunk), z_dim, t_chunk, max_seq_len)).to(device)
            T = torch.tensor(t_chunk, dtype=torch.long)
            X_hat = rec(sup(gen(Z, T), T), T).cpu().numpy()
            for i, t_i in enumerate(t_chunk):
                # Recortar padding y desnormalizar
                generated_data.append(X_hat[i, :t_i, :] * max_val + min_val)

    return generated_data


# ---------------------------------------------------------------------------
# Ejemplo con datos senoidales (como sine_data_generation del repo oficial)
# ---------------------------------------------------------------------------
def sine_data_generation(no, seq_len, dim):
    data = []
    for _ in range(no):
        temp = []
        for _ in range(dim):
            freq = np.random.uniform(0, 0.1)
            phase = np.random.uniform(0, 0.1)
            temp.append([np.sin(freq * j + phase) for j in range(seq_len)])
        temp = np.transpose(np.asarray(temp))
        data.append((temp + 1) * 0.5)
    return np.asarray(data, dtype=np.float32)


if __name__ == "__main__":
    data = sine_data_generation(no=1000, seq_len=24, dim=5)
    params = {
        "hidden_dim": 24,
        "num_layer": 3,
        "iterations": 10000,
        "batch_size": 128,
        "module": "gru",
    }
    synthetic = timegan(data, params)
    print(len(synthetic), synthetic[0].shape)