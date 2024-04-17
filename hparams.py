hparams = {
    "model": {
        "dx": 50,
        "dy": 3,
        "L": 100, 
        "num_heads": 3,
        "use_bias": False,
        "dropout_rate": 0.0,
        "attention_type": "softmax",
        "q_k_v_o_proj_enabled": [True, True, True, True],
    }, 
    "data": {
        "number_of_samples": 10000,
        "noise_std": 0.0,
    },
    "train": {
        "optimizer": "Adam",
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "momentum": 0.0,
        "batch_size": 64,
        "max_steps": 100_000,
        "warmup_steps": 2_000,
        "lr_decay_steps": 100_000,
        "min_lr": 5e-5,
        "use_stored_data": False,
        "use_logger": True,
        "use_mixed_precision": True, 
        "num_epochs": 1000000,
        "print_every_n_epochs": 1,
        "visualize_every_n_epochs": 10,
        "initialize_weights": .01, # lambda
        "max_learning_rate": 5e-1,
        "min_learning_rate": 1e-4,
        "learning_rate": 1e-3,
        "grad_elbow": .1,
        "lr_schedule_method": "constant", # "elbow_decay"
    }
}
