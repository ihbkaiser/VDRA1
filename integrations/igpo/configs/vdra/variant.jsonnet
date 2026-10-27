// Merge VDRA_VARIANT_JSON (e.g. '{"proxy":"constant"}') over the VDRA defaults.
{
  episode_generator+: {
    inference_strategy+: {
      vdra+: std.parseJson(std.extVar('APP_VDRA_VARIANT_JSON')),
    },
  },
}
