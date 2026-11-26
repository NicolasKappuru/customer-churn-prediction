"""Wasserstein TimeGAN (WGAN-GP) en PyTorch.

Misma arquitectura y mismas fases 1 y 2 que ``TimeGAN.timegan``; sólo cambia la
parte adversarial de la fase conjunta:

- El discriminador pasa a ser un **crítico**: da un puntaje sin sigmoide (la red
  ya no tenía activación de salida). Puntaje de un usuario = promedio de su
  salida por día.
- Pérdida del crítico (Gulrajani et al., 2017, Algoritmo 1)::

      L_D = [c(H^) - c(H)] + gamma [c(E^) - c(H)] + lambda_gp (GP(H, H^) + gamma GP(H, E^))

  donde ``c(H) - c(H^)`` estima la distancia de Wasserstein entre reales y
  sintéticos, y ``GP`` = ``E[(||grad c(mezcla)|| - 1)^2]`` sobre mezclas
  aleatorias entre un usuario real y uno sintético (restricción de Lipschitz).
- Pérdida adversarial del generador: ``-c(H^) - gamma c(E^)``. Las pérdidas
  supervisada (``100 sqrt(L_S)``) y de momentos (``100 L_V``) no cambian.
- ``n_critic`` pasos del crítico por cada paso del generador (el TimeGAN
  original hacía lo contrario: 2 del generador por 1 del discriminador).
- Desaparece la regla ``D_loss > 0.15`` del código oficial de TimeGAN: el
  crítico se entrena siempre.
- Adam con ``lr=1e-4, betas=(0.5, 0.9)`` para crítico y generador, como en
  WGAN-GP (``g_lr`` / ``d_lr`` los separan). El autoencoder conserva los
  optimizadores de TimeGAN.

Crítico (``critic``):

- ``"gru"`` (por defecto, v1): el discriminador recurrente de TimeGAN. cuDNN no
  implementa la doble derivada de sus RNN, que la penalización de gradiente
  necesita; ese cálculo corre con cuDNN apagado (~11x más lento por paso).
- ``"tcn"`` (v2): ``TCNCritic``, convoluciones 1D dilatadas no causales sobre la
  secuencia latente. Soporta la doble derivada con cuDNN. TimeGAN no restringe la
  arquitectura del discriminador (Yoon et al., 2019) y WGAN-GP usa críticos
  convolucionales 1D para secuencias (Gulrajani et al., 2017; Donahue et al., 2019).

El checkpoint tiene el mismo formato que el de TimeGAN (más la clave ``critic``);
``TimeGAN.generate_from_checkpoint`` sirve para ambos: generar no usa el crítico.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from TimeGAN import (TimeGAN, MinMaxScaler, _generate, batch_generator,
                     extract_time, random_generator)


class TCNCritic(nn.Module):
    """Crítico convolucional temporal: ``H (B, L, hidden) -> puntaje por día (B, L, 1)``.

    Convoluciones 1D dilatadas (kernel 3, dilataciones 1, 2, 4, ..., 32) con
    LeakyReLU y sin normalización por batch (rompería la penalización de
    gradiente, que es por usuario). No es causal: cada salida ve la secuencia en
    ambas direcciones, como el discriminador bidireccional del paper de TimeGAN.
    Campo receptivo = ``1 + (kernel - 1) * sum(dilations)`` = 127 días >= 90.
    Misma interfaz que ``RNNNet``: ``forward(H, T)``; los días de padding se ponen
    en cero a la entrada.
    """

    def __init__(self, hidden_dim, channels=64, kernel_size=3,
                 dilations=(1, 2, 4, 8, 16, 32)):
        super().__init__()
        layers, c_in = [], hidden_dim
        for d in dilations:
            layers += [nn.Conv1d(c_in, channels, kernel_size, dilation=d,
                                 padding=d * (kernel_size - 1) // 2),
                       nn.LeakyReLU(0.2)]
            c_in = channels
        self.net = nn.Sequential(*layers)
        self.out = nn.Conv1d(channels, 1, 1)
        self.receptive_field = 1 + (kernel_size - 1) * sum(dilations)

    def forward(self, H, T):
        mask = (torch.arange(H.size(1), device=H.device)[None, :]
                < T.to(H.device)[:, None]).unsqueeze(-1).to(H.dtype)
        x = (H * mask).transpose(1, 2)                                   # (B, hidden, L)
        return self.out(self.net(x)).transpose(1, 2)                     # (B, L, 1)


def _weight_norm(module):
    return float(torch.sqrt(sum((p.detach() ** 2).sum() for p in module.parameters())))


def wtimegan(ori_data, parameters):
    """Entrena el Wasserstein TimeGAN y devuelve datos sintéticos.

    Mismos argumentos y salida que ``TimeGAN.timegan``. Parámetros extra
    (opcionales):

      joint_iterations  iteraciones de la fase conjunta (por defecto
                 ``iterations``). Cada una hace ``n_critic`` pasos del crítico,
                 así que ``iterations // n_critic`` da tantos pasos del crítico
                 como pasos del discriminador hace TimeGAN con ``iterations``.
      n_critic   pasos del crítico por iteración conjunta (por defecto 5)
      lambda_gp  peso de la penalización de gradiente (por defecto 10)
      gan_lr     learning rate de crítico y generador (por defecto 1e-4)
      g_lr       learning rate del generador + supervisor (por defecto ``gan_lr``)
      d_lr       learning rate del crítico (por defecto ``gan_lr``)
      gan_betas  betas de Adam de crítico y generador (por defecto [0.5, 0.9])
      adv_weight peso de la pérdida adversarial del generador frente a
                 ``100 sqrt(L_S) + 100 L_V`` (por defecto 1)
      critic     ``"gru"`` (por defecto) o ``"tcn"`` (``TCNCritic``)
      critic_channels  canales de cada capa de ``TCNCritic`` (por defecto 64)

    El historial de la fase conjunta incluye ``g_weight_norm``, la norma de los
    pesos del generador: si casi no se aleja de su valor inicial, el generador
    avanza poco con el presupuesto dado (lo que limitó a la v1).
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
    log_every = parameters.get("log_every", 100)
    joint_iterations = parameters.get("joint_iterations", iterations)
    n_critic = parameters.get("n_critic", 5)
    lambda_gp = parameters.get("lambda_gp", 10.0)
    gan_lr = parameters.get("gan_lr", 1e-4)
    gan_betas = tuple(parameters.get("gan_betas", (0.5, 0.9)))
    adv_weight = parameters.get("adv_weight", 1.0)
    g_lr = parameters.get("g_lr", gan_lr)
    d_lr = parameters.get("d_lr", gan_lr)
    critic_arch = parameters.get("critic", "gru")
    if critic_arch not in ("gru", "tcn"):
        raise ValueError(f"critic must be 'gru' or 'tcn', got {critic_arch!r}")
    history = []

    model = TimeGAN(dim, hidden_dim, num_layers, module_name)
    if critic_arch == "tcn":
        model.discriminator = TCNCritic(hidden_dim, channels=parameters.get("critic_channels", 64))
    model.to(device)
    emb, rec, gen, sup, critic = (model.embedder, model.recovery, model.generator,
                                  model.supervisor, model.discriminator)
    # Sólo la GRU necesita apagar cuDNN para la doble derivada
    gp_cudnn = critic_arch != "gru"

    er_params = list(emb.parameters()) + list(rec.parameters())
    gs_params = list(gen.parameters()) + list(sup.parameters())

    # Fases 1 y 2 y el autoencoder en la fase 3: igual que TimeGAN
    E0_solver = torch.optim.Adam(er_params)
    E_solver = torch.optim.Adam(er_params)
    GS_solver = torch.optim.Adam(gs_params)
    # Parte adversarial: hiperparámetros de WGAN-GP
    D_solver = torch.optim.Adam(critic.parameters(), lr=d_lr, betas=gan_betas)
    G_solver = torch.optim.Adam(gs_params, lr=g_lr, betas=gan_betas)

    # --- helpers ---------------------------------------------------------
    def sample_batch(with_z=False):
        X_mb, T_mb = batch_generator(ori_data, ori_time, batch_size)
        X = torch.from_numpy(X_mb).to(device)
        T = torch.tensor(T_mb, dtype=torch.long)
        if not with_z:
            return X, T
        Z = torch.from_numpy(random_generator(len(T_mb), z_dim, T_mb, max_seq_len)).to(device)
        return X, T, Z

    def critic_score(H, T):
        """Puntaje por usuario: promedio de la salida del crítico en sus días válidos."""
        out = critic(H, T).squeeze(-1)                                   # (B, L)
        mask = (torch.arange(out.size(1), device=device)[None, :]
                < T.to(device)[:, None]).float()
        return (out * mask).sum(dim=1) / mask.sum(dim=1)

    def gradient_penalties(H_real, fakes, T):
        """E[(||grad c(x_mix)|| - 1)^2] por cada sintético de ``fakes``,
        con x_mix = eps H_real + (1 - eps) H_fake.

        Todas las mezclas van en una sola pasada del crítico: sin cuDNN la GRU
        cuesta lo mismo con 128 que con 256 usuarios (recorre los días uno a uno).
        """
        n = H_real.size(0)
        eps = torch.rand(n * len(fakes), 1, 1, device=device)
        real = H_real.repeat(len(fakes), 1, 1)
        mix = (eps * real + (1 - eps) * torch.cat(fakes)).requires_grad_(True)
        # cuDNN no soporta la doble derivada de la GRU (la TCN sí)
        with torch.backends.cudnn.flags(enabled=gp_cudnn):
            scores = critic_score(mix, T.repeat(len(fakes)))
            grad, = torch.autograd.grad(scores.sum(), mix, create_graph=True)
        penalty = (grad.flatten(1).norm(2, dim=1) - 1) ** 2
        return penalty.view(len(fakes), n).mean(dim=1)

    def supervised_loss(H, T):
        H_hat_supervise = sup(H, T)
        return F.mse_loss(H_hat_supervise[:, :-1, :], H[:, 1:, :])

    def step(loss, optimizer):
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
        if itt % log_every == 0:
            history.append({"phase": "embedding", "iteration": itt,
                            "e_loss_t0": float(np.sqrt(E_loss_T0.item()))})
        if itt % 1000 == 0:
            print(f"step: {itt}/{iterations}, e_loss: {np.sqrt(E_loss_T0.item()):.4f}")
    print("Finish Embedding Network Training")

    # --- 2. Solo pérdida supervisada (supervisor) --------------------------
    print("Start Training with Supervised Loss Only")
    for itt in range(iterations):
        X, T = sample_batch()
        with torch.no_grad():
            H = emb(X, T)
        G_loss_S = supervised_loss(H, T)
        step(G_loss_S, GS_solver)
        if itt % log_every == 0:
            history.append({"phase": "supervised", "iteration": itt,
                            "g_loss_s": float(np.sqrt(G_loss_S.item()))})
        if itt % 1000 == 0:
            print(f"step: {itt}/{iterations}, s_loss: {np.sqrt(G_loss_S.item()):.4f}")
    print("Finish Training with Supervised Loss Only")

    # --- 3. Entrenamiento conjunto (Wasserstein) ---------------------------
    print(f"Start Joint Training (WGAN-GP, critic={critic_arch}, g_lr={g_lr}, d_lr={d_lr})")
    g_norm_start = _weight_norm(gen)
    for itt in range(joint_iterations):
        # Crítico: n_critic pasos, siempre (sin la regla D_loss > 0.15)
        for _ in range(n_critic):
            X, T, Z = sample_batch(with_z=True)
            with torch.no_grad():
                H = emb(X, T)
                E_hat = gen(Z, T)
                H_hat = sup(E_hat, T)
            # Real, H^ y E^ en una sola pasada del crítico
            c_real, c_hat, c_e = critic_score(
                torch.cat([H, H_hat, E_hat]), T.repeat(3)).view(3, -1).mean(dim=1)
            gp_hat, gp_e = gradient_penalties(H, [H_hat, E_hat], T)
            gp = gp_hat + gamma * gp_e
            D_loss = (c_hat - c_real) + gamma * (c_e - c_real) + lambda_gp * gp
            step(D_loss, D_solver)
        w_distance = (c_real - c_hat).item()   # estimación de W(real, H^) del último paso

        # Generador + supervisor: un paso
        X, T, Z = sample_batch(with_z=True)
        H = emb(X, T).detach()
        E_hat = gen(Z, T)
        H_hat = sup(E_hat, T)
        X_hat = rec(H_hat, T)

        G_loss_U = -critic_score(H_hat, T).mean()
        G_loss_U_e = -critic_score(E_hat, T).mean()
        G_loss_S = supervised_loss(H, T)
        G_loss_V1 = torch.mean(torch.abs(
            torch.sqrt(X_hat.var(dim=0, unbiased=False) + 1e-6)
            - torch.sqrt(X.var(dim=0, unbiased=False) + 1e-6)))
        G_loss_V2 = torch.mean(torch.abs(X_hat.mean(dim=0) - X.mean(dim=0)))
        G_loss_V = G_loss_V1 + G_loss_V2

        G_loss = (adv_weight * (G_loss_U + gamma * G_loss_U_e)
                  + 100 * torch.sqrt(G_loss_S) + 100 * G_loss_V)
        step(G_loss, G_solver)

        # Embedder + recovery: un paso (igual que TimeGAN)
        H = emb(X, T)
        X_tilde = rec(H, T)
        E_loss_T0 = F.mse_loss(X_tilde, X)
        E_loss = 10 * torch.sqrt(E_loss_T0) + 0.1 * supervised_loss(H, T)
        step(E_loss, E_solver)

        if itt % log_every == 0:
            history.append({
                "phase": "joint", "iteration": itt,
                "d_loss": D_loss.item(), "w_distance": w_distance,
                "gradient_penalty": gp.item(),
                "g_loss_u": G_loss_U.item(), "g_loss_u_e": G_loss_U_e.item(),
                "g_loss_s": float(np.sqrt(G_loss_S.item())),
                "g_loss_v": G_loss_V.item(),
                "e_loss_t0": float(np.sqrt(E_loss_T0.item())),
                "g_weight_norm": _weight_norm(gen),
            })
        if itt % max(1, joint_iterations // 10) == 0:
            print(f"step: {itt}/{joint_iterations}, d_loss: {D_loss.item():.4f}, "
                  f"w_dist: {w_distance:.4f}, gp: {gp.item():.4f}, "
                  f"g_loss_u: {G_loss_U.item():.4f}, "
                  f"g_loss_s: {np.sqrt(G_loss_S.item()):.4f}, "
                  f"g_loss_v: {G_loss_V.item():.4f}, "
                  f"e_loss_t0: {np.sqrt(E_loss_T0.item()):.4f}, "
                  f"|G|: {g_norm_start:.1f} -> {_weight_norm(gen):.1f}")
    print("Finish Joint Training")

    # --- Generación de datos sintéticos ------------------------------------
    n_samples = parameters.get("n_samples", no)
    gen_time = [ori_time[i] for i in np.random.randint(0, no, n_samples)]
    generated_data = _generate(model, gen_time, z_dim, max_seq_len, batch_size,
                               min_val, max_val, device)

    if not parameters.get("return_artifacts", False):
        return generated_data
    checkpoint = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "dim": dim, "hidden_dim": hidden_dim, "num_layer": num_layers,
        "module": module_name, "min_val": min_val, "max_val": max_val,
        "ori_time": ori_time, "max_seq_len": max_seq_len, "loss": "wgan-gp",
        "critic": critic_arch,
    }
    return generated_data, {"history": history, "checkpoint": checkpoint}
