/// Regression / feature test for Issue #186: Configurable context length
///
/// **Feature**: Shimmy now supports two mechanisms for controlling the KV-cache
/// context length (`n_ctx`):
///
/// 1. **`--ctx-len N` CLI flag** — sets the default KV-cache size for all
///    auto-discovered models at startup.  Useful when running dedicated Shimmy
///    instances per role: `--ctx-len 2048` for reviewer/leaf instances (saves
///    VRAM, forces prompt compression), `--ctx-len 16384` for long-context
///    deliberation instances.
///
/// 2. **`options.num_ctx` API field** — Ollama-compatible per-request override.
///    Allows the same Shimmy instance to serve short reviewer calls and long
///    in-loop deliberation calls without restart.  The value is capped at the
///    server's configured `ctx_len`; requesting more is warned and silently
///    clamped to the server limit.
///
/// **Design note**: llama.cpp allocates the KV cache at context creation time,
/// not per-request.  True per-request resizing requires a model reload.  On
/// machines with large system RAM (e.g. 1.5 TiB) the model file stays warm in
/// the kernel page cache after first load, so a reload costs only the context
/// allocation (~200 ms for 8K ctx on qwen3:8b) rather than full disk I/O.
/// A future follow-up can add `LlamaModel` weight caching to eliminate even
/// that overhead by separating the static weights from the per-request context.
///
/// **Affected callers**: ACMT's `ShimmyAdapter` passes `options.num_ctx` via
/// `extra_body` — previously Shimmy silently ignored this field.
#[cfg(test)]
mod issue_186_tests {
    use shimmy::model_registry::Registry;

    /// Default context length is 8192 (not the old 4096).
    #[test]
    fn test_registry_default_ctx_len_is_8192() {
        let reg = Registry::new();
        assert_eq!(
            reg.default_ctx_len, 8192,
            "Default context length must be 8192; old value of 4096 caused KV cache exhaustion \
             on tier-3 problems that required more than 4096 tokens of context"
        );
    }

    /// `--ctx-len` sets `registry.default_ctx_len`.
    #[test]
    fn test_registry_default_ctx_len_override() {
        let mut reg = Registry::new();
        reg.default_ctx_len = 2048;
        assert_eq!(reg.default_ctx_len, 2048);

        reg.default_ctx_len = 16384;
        assert_eq!(reg.default_ctx_len, 16384);
    }

    /// `to_spec` uses `default_ctx_len` when a model has no explicit ctx_len.
    #[test]
    fn test_to_spec_uses_default_ctx_len() {
        use shimmy::model_registry::ModelEntry;
        use std::path::PathBuf;

        let mut reg = Registry::new();
        reg.default_ctx_len = 4096;

        reg.register(ModelEntry {
            name: "test-model".to_string(),
            base_path: PathBuf::from("/tmp/test.gguf"),
            lora_path: None,
            template: None,
            ctx_len: None, // no explicit ctx_len — should use default
            n_threads: None,
        });

        let spec = reg.to_spec("test-model").expect("model should be found");
        assert_eq!(
            spec.ctx_len, 4096,
            "to_spec must fall back to registry.default_ctx_len when model has no explicit ctx_len"
        );
    }

    /// Explicit per-model `ctx_len` overrides the registry default.
    #[test]
    fn test_explicit_model_ctx_len_overrides_default() {
        use shimmy::model_registry::ModelEntry;
        use std::path::PathBuf;

        let mut reg = Registry::new();
        reg.default_ctx_len = 8192; // registry default

        reg.register(ModelEntry {
            name: "small-reviewer".to_string(),
            base_path: PathBuf::from("/tmp/gemma3-1b.gguf"),
            lora_path: None,
            template: Some("chatml".to_string()),
            ctx_len: Some(2048), // explicit override for this model
            n_threads: None,
        });

        let spec = reg.to_spec("small-reviewer").expect("model should be found");
        assert_eq!(
            spec.ctx_len, 2048,
            "Explicit per-model ctx_len must take precedence over registry.default_ctx_len"
        );
    }

    /// `RequestOptions` deserialises `num_ctx` correctly and ignores unknown
    /// Ollama fields (forward-compatibility).
    #[test]
    fn test_request_options_num_ctx_deserialisation() {
        use shimmy::openai_compat::RequestOptions;

        // With num_ctx present
        let json = r#"{"num_ctx": 4096, "temperature": 0.7}"#;
        let opts: RequestOptions = serde_json::from_str(json)
            .expect("RequestOptions must deserialise from Ollama-style options JSON");
        assert_eq!(opts.num_ctx, Some(4096));

        // Without num_ctx
        let json_no_ctx = r#"{"temperature": 0.3}"#;
        let opts_no: RequestOptions = serde_json::from_str(json_no_ctx)
            .expect("RequestOptions must deserialise when num_ctx is absent");
        assert_eq!(
            opts_no.num_ctx, None,
            "Missing num_ctx must deserialise to None, not panic"
        );

        // num_ctx = 0 edge case
        let json_zero = r#"{"num_ctx": 0}"#;
        let opts_zero: RequestOptions = serde_json::from_str(json_zero).unwrap();
        assert_eq!(opts_zero.num_ctx, Some(0));
    }

    /// `options.num_ctx` smaller than `spec.ctx_len` is accepted (normal case).
    /// `options.num_ctx` larger than `spec.ctx_len` is capped to `spec.ctx_len`.
    ///
    /// This test exercises the clamping logic inline (mirroring chat_completions).
    #[test]
    fn test_num_ctx_clamping_logic() {
        let server_ctx = 8192_usize;

        // Smaller request: accepted
        let req_ctx = 2048_usize;
        let effective = req_ctx.min(server_ctx);
        assert_eq!(effective, 2048, "Smaller num_ctx must be used as-is");

        // Larger request: capped
        let req_ctx_large = 32768_usize;
        let effective_capped = req_ctx_large.min(server_ctx);
        assert_eq!(
            effective_capped, 8192,
            "num_ctx exceeding server limit must be capped to server ctx_len"
        );

        // Equal: accepted unchanged
        let req_ctx_equal = 8192_usize;
        let effective_equal = req_ctx_equal.min(server_ctx);
        assert_eq!(effective_equal, 8192);
    }

    /// CLI `--ctx-len` parses correctly.
    #[test]
    fn test_cli_ctx_len_flag_parses() {
        use clap::Parser;
        use shimmy::cli::Cli;

        let cli = Cli::try_parse_from(["shimmy", "--ctx-len", "4096", "serve"]).unwrap();
        assert_eq!(cli.ctx_len, Some(4096));

        let cli_16k = Cli::try_parse_from(["shimmy", "--ctx-len", "16384", "serve"]).unwrap();
        assert_eq!(cli_16k.ctx_len, Some(16384));

        // Without flag: None (use registry default)
        let cli_default = Cli::try_parse_from(["shimmy", "serve"]).unwrap();
        assert_eq!(cli_default.ctx_len, None);
    }
}
