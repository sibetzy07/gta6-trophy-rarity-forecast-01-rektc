# gta6-trophy-rarity-forecast-01-rektc
[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sibetzy07/gta6-trophy-rarity-forecast-01-rektc/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sibetzy07/gta6-trophy-rarity-forecast-01-rektc/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A forecaster for every Grand Theft Auto VI trophy's earn rate on Exophase in week one. It models earners and players-tracked separately — new players dilute every rate while players earn trophies over time — stepping the earners forward with a hazard that decays since launch, so a stale launch-night page forecasts the refresh's drop and a moving page forecasts progression against continued dilution, with a wide honest spread on the logit scale.

## Run it

From a directory holding the kit's `input/`:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

It reads every `*.in` case in `input/` and writes one `<id>.out` per case to `out/`: five quantiles (5/25/50/75/95%) of each trophy's earn rate at the case's target. Node 22, standard library only, no network, fully deterministic.

<sub>Entered on [make.fun](https://make.fun/?ref=vrw8s2cx&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
