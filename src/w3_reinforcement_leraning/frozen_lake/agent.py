"""Train and visualize a tabular Q-learning agent on FrozenLake."""

import argparse
import json
import random
from pathlib import Path

from env import DEFAULT_MAP_NAME, make_env


DEFAULT_Q_TABLE = Path("models") / "frozenlake_qtable.json"


def choose_action(
    q_table: list[list[float]],
    state: int,
    epsilon: float,
    rng: random.Random,
) -> int:
    """Choose an epsilon-greedy action, breaking ties randomly."""
    if rng.random() < epsilon:
        return rng.randrange(len(q_table[state]))

    values = q_table[state]
    best_value = max(values)
    best_actions = [action for action, value in enumerate(values) if value == best_value]
    return rng.choice(best_actions)


def save_q_table(
    path: Path,
    q_table: list[list[float]],
    map_name: str,
    slippery: bool,
) -> None:
    """Persist the learned values and environment settings as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "map_name": map_name,
                "slippery": slippery,
                "q_table": q_table,
            },
            file,
            indent=2,
        )


def load_q_table(path: Path) -> tuple[list[list[float]], str, bool]:
    """Load a saved Q-table and its matching FrozenLake settings."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    q_table = data["q_table"]
    map_name = data["map_name"]
    slippery = data["slippery"]
    if not isinstance(q_table, list) or not q_table or not all(
        isinstance(row, list) and row for row in q_table
    ):
        raise ValueError(f"Invalid Q-table in {path}")
    return q_table, map_name, slippery


def render_greedy_policy(
    q_table: list[list[float]],
    map_name: str,
    slippery: bool,
    seed: int | None,
    max_steps: int,
) -> float:
    """Run one greedy episode in a visible window and return its total reward."""
    env = make_env(render_mode="human", map_name=map_name, slippery=slippery)
    rng = random.Random(seed)
    try:
        state, _ = env.reset(seed=seed)
        total_reward = 0.0
        for _ in range(max_steps):
            action = choose_action(q_table, state, epsilon=0.0, rng=rng)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return total_reward
    finally:
        env.close()


def train(args: argparse.Namespace) -> None:
    """Train the Q-table and optionally show greedy-policy snapshots."""
    rng = random.Random(args.seed)
    env = make_env(map_name=args.map_name, slippery=not args.deterministic)
    q_table = [
        [0.0] * env.action_space.n for _ in range(env.observation_space.n)
    ]
    recent_successes: list[bool] = []

    try:
        for episode in range(args.episodes):
            state, _ = env.reset(seed=args.seed if episode == 0 else None)
            epsilon = max(
                args.epsilon_end,
                args.epsilon_start
                - (args.epsilon_start - args.epsilon_end)
                * episode
                / max(args.episodes - 1, 1),
            )
            episode_reward = 0.0

            for step in range(args.max_steps):
                action = choose_action(q_table, state, epsilon, rng)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                finished = terminated or truncated or step == args.max_steps - 1

                best_next_value = max(q_table[next_state])
                target = reward if finished else reward + args.gamma * best_next_value
                q_table[state][action] += args.alpha * (
                    target - q_table[state][action]
                )
                state = next_state
                if finished:
                    break

            recent_successes.append(episode_reward > 0)
            if len(recent_successes) > args.report_every:
                recent_successes.pop(0)

            if (episode + 1) % args.report_every == 0 or episode + 1 == args.episodes:
                success_rate = sum(recent_successes) / len(recent_successes)
                print(
                    f"Episode {episode + 1}/{args.episodes} | "
                    f"epsilon={epsilon:.3f} | "
                    f"success rate (last {len(recent_successes)}): "
                    f"{success_rate:.1%}"
                )

            if args.render_every and (episode + 1) % args.render_every == 0:
                reward = render_greedy_policy(
                    q_table,
                    args.map_name,
                    not args.deterministic,
                    args.seed,
                    args.max_steps,
                )
                print(
                    f"Rendered policy after episode {episode + 1}: "
                    f"{'goal reached' if reward > 0 else 'goal not reached'}"
                )
    finally:
        env.close()

    save_q_table(
        args.q_table,
        q_table,
        args.map_name,
        not args.deterministic,
    )
    print(f"Q-table saved to: {args.q_table}")


def play(args: argparse.Namespace) -> None:
    """Load a saved Q-table and demonstrate the learned policy."""
    q_table, map_name, slippery = load_q_table(args.q_table)
    env = make_env(
        render_mode="human" if not args.no_render else None,
        map_name=map_name,
        slippery=slippery,
    )
    rng = random.Random(args.seed)

    try:
        for episode in range(args.episodes):
            state, _ = env.reset(seed=args.seed if episode == 0 else None)
            total_reward = 0.0
            for _ in range(args.max_steps):
                action = choose_action(q_table, state, epsilon=0.0, rng=rng)
                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            print(
                f"Episode {episode + 1}/{args.episodes}: "
                f"{'goal reached' if total_reward > 0 else 'goal not reached'}"
            )
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FrozenLake Q-learning example")
    commands = parser.add_subparsers(dest="command", required=True)

    train_parser = commands.add_parser("train", help="train and save a Q-table")
    train_parser.add_argument("--episodes", type=int, default=5000)
    train_parser.add_argument("--max-steps", type=int, default=100)
    train_parser.add_argument("--alpha", type=float, default=0.8)
    train_parser.add_argument("--gamma", type=float, default=0.95)
    train_parser.add_argument("--epsilon-start", type=float, default=1.0)
    train_parser.add_argument("--epsilon-end", type=float, default=0.05)
    train_parser.add_argument("--report-every", type=int, default=100)
    train_parser.add_argument(
        "--render-every",
        type=int,
        default=0,
        help="show the current greedy policy every N training episodes (0 disables it)",
    )
    train_parser.add_argument("--map-name", default=DEFAULT_MAP_NAME)
    train_parser.add_argument("--deterministic", action="store_true")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--q-table", type=Path, default=DEFAULT_Q_TABLE)
    train_parser.set_defaults(func=train)

    play_parser = commands.add_parser("play", help="visualize a saved Q-table")
    play_parser.add_argument("--episodes", type=int, default=3)
    play_parser.add_argument("--max-steps", type=int, default=100)
    play_parser.add_argument("--seed", type=int, default=42)
    play_parser.add_argument("--q-table", type=Path, default=DEFAULT_Q_TABLE)
    play_parser.add_argument(
        "--no-render",
        action="store_true",
        help="run without opening the FrozenLake window",
    )
    play_parser.set_defaults(func=play)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    arguments = parser.parse_args()
    if arguments.episodes < 1 or arguments.max_steps < 1:
        parser.error("--episodes and --max-steps must be greater than zero")
    if arguments.command == "train":
        if arguments.report_every < 1 or arguments.render_every < 0:
            parser.error(
                "--report-every must be positive and --render-every cannot be negative"
            )
        if not 0 < arguments.alpha <= 1 or not 0 <= arguments.gamma <= 1:
            parser.error("--alpha must be in (0, 1] and --gamma must be in [0, 1]")
        if not 0 <= arguments.epsilon_end <= arguments.epsilon_start <= 1:
            parser.error("epsilon values must satisfy 0 <= end <= start <= 1")
    arguments.func(arguments)