import hashlib

def get_pinch_controllers_save_path(args):
    save_path = "controllers/pinch_controllers/"
    save_path += f"{args.env_id}/"

    config_str = f"alg={args.algo}\n"
    config_str += f"n1={args.n1}\n"
    config_str += f"n2={args.n2}\n"
    config_str += f"env_cnt={args.env_cnt}\n"
    config_str += f"timesteps={args.timesteps}\n"
    config_str += f"seed={args.seed}\n"
    config_str += f"lr={args.lr}\n"
    config_str += f"eval_every={args.eval_every}\n"
    config_str += f"eval_episodes={args.timesteps}\n"
    config_str += f"seed={args.seed}\n"
    config_str += f"gamma={args.gamma}\n"
    config_str += f"reward_norm_flag={args.reward_norm_flag}\n"
    config_str += f"obs_norm_flag={args.obs_norm_flag}"

    save_path += "config_" + hashlib.sha1(config_str.encode("UTF-8")).hexdigest()
    
    return save_path, config_str    

def get_trained_rl_controller_save_path(args):
    save_path = "controllers/base_controllers/"
    save_path += f"{args.env_id}/"

    config_str = f"alg={args.algo}\n"
    config_str += f"n1={args.n1}\n"
    config_str += f"n2={args.n2}\n"
    config_str += f"env_cnt={args.env_cnt}\n"
    config_str += f"timesteps={args.timesteps}\n"
    config_str += f"seed={args.seed}\n"
    config_str += f"lr={args.lr}\n"
    config_str += f"eval_every={args.eval_every}\n"
    config_str += f"eval_episodes={args.timesteps}\n"
    config_str += f"seed={args.seed}\n"
    config_str += f"gamma={args.gamma}\n"
    config_str += f"reward_norm_flag={args.reward_norm_flag}\n"
    config_str += f"obs_norm_flag={args.obs_norm_flag}"

    save_path += "config_" + hashlib.sha1(config_str.encode("UTF-8")).hexdigest()
    
    return save_path, config_str

def get_dataset_save_path(args):
    save_path = "env_data/"
    save_path += f"{args.env_id}/"

    config_str = f"alg={args.algo}\n"
    config_str += f"n1={args.n1}\n"
    config_str += f"n2={args.n2}\n"
    config_str += f"env_cnt={args.env_cnt}\n"
    config_str += f"timesteps={args.timesteps}\n"
    config_str += f"seed={args.seed}\n"
    config_str += f"lr={args.lr}\n"
    config_str += f"eval_every={args.eval_every}\n"
    config_str += f"eval_episodes={args.timesteps}\n"
    config_str += f"seed={args.seed}\n"
    config_str += f"gamma={args.gamma}\n"
    config_str += f"reward_norm_flag={args.reward_norm_flag}\n"
    config_str += f"obs_norm_flag={args.obs_norm_flag}\n"
    config_str += f"sample_cnt={args.sample_cnt}"

    save_path += "config_" + hashlib.sha1(config_str.encode("UTF-8")).hexdigest()

    return save_path, config_str

def get_latent_controller_path(args):
    save_path = "controllers/im_controllers/"
    save_path += f"{args.env_id}/"

    config_str = f"alg={args.algo}\n"
    config_str += f"n1={args.n1}\n"
    config_str += f"n2={args.n2}\n"    
    config_str += f"env_cnt={args.env_cnt}\n"
    config_str += f"timesteps={args.timesteps}\n"
    config_str += f"seed={args.seed}\n"
    config_str += f"lr={args.lr}\n"
    config_str += f"eval_every={args.eval_every}\n"
    config_str += f"eval_episodes={args.timesteps}\n"
    config_str += f"seed={args.seed}\n"
    config_str += f"gamma={args.gamma}\n"
    config_str += f"reward_norm_flag={args.reward_norm_flag}\n"
    config_str += f"obs_norm_flag={args.obs_norm_flag}\n"
    config_str += f"sample_cnt={args.sample_cnt}\n"
    config_str += f"encoder={args.encoder}\n"
    config_str += f"latent_dim={args.latent_dim}\n"
    config_str += f"latent_hidden={args.latent_hidden}\n"
    config_str += f"controller_hidden={args.controller_hidden}\n"
    config_str += f"epochs={args.epochs}\n"
    config_str += f"bs={args.bs}\n"
    config_str += f"latent_lr={args.latent_lr}"

    save_path += "config_" + hashlib.sha1(config_str.encode("UTF-8")).hexdigest()

    return save_path, config_str