"""Stage5 refinement decoder 测试入口。

复用已验证的 idea3 A-F 测试逻辑；Stage5 checkpoint 由命令行传入，
模型加载流程会自动激活三组 UNet LoRA 和 VAE encoder LoRA adapter。
"""

from test_idea3_decoder_refinement import main, parse_args


if __name__ == "__main__":
    main(parse_args())
