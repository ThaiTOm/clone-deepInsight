from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.margin_list = (1.0, 0.5, 0.0)
config.network = "r50"
config.resume = False
config.output = None
config.embedding_size = 1024
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 16
config.lr = 0.01
config.verbose = 10
config.dali = False

config.rec = "/train_tmp/ms1m-retinaface-t1"
config.num_classes = 304872
config.num_image = 2606980
config.num_epoch = 20
config.warmup_epoch = 0
config.val_targets = ['lfw', 'cfp_fp', "agedb_30"]
config.using_wandb = True
config.wandb_key = "8b14dfbe9ea4cbd166817a01fdeccc6dfc3089dc"
config.wandb_resume = False
config.wandb_log_all = True
config.wandb_entity = "nguyenduythai"
config.wandb_project = "FirstTime"
config.num_workers = 1
config.frequent = 50
config.save_all_states = True
config.resume = True
config.kaggle_dir = True