from train_stage1_lle_control import parse_args, train_main


if __name__ == "__main__":
    train_main(parse_args(stage_default="stage2"))
