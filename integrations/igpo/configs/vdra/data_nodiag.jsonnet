// Train on the MATH corpus with the 256-problem diagnostic split removed
// (created by scripts/01_prepare_data.sh).
{
  episode_generator+: {
    task+: {
      dataset_dict_path: std.extVar('APP_VDRA_TRAIN_DATASET_DICT'),
    },
  },
}
