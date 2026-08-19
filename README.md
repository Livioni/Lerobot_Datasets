# Rerun Lerobot 3.0


## Launch

```bash
conda run -n rerun python visualize_lerobot_rerun.py \
  --dataset LIBERO/meta \
  --episode 0
```

## Export one MP4 per episode

Export every episode from every task dataset into
`visualization/<dataset>/episode_XXXXXX.mp4`:

```bash
conda run -n rerun python export_lerobot_episode_mp4.py
```

Test a small selection first:

```bash
conda run -n rerun python export_lerobot_episode_mp4.py \
  --dataset task3_bimanual_plug_Insertion \
  --episodes 0-2
```

Existing MP4s are skipped, so the command can be resumed. Use `--overwrite` to
replace them, `--jobs 2` to process two episodes concurrently, or `--mode both`
to additionally export every camera as its own MP4. Run with `--help` for all
options.
