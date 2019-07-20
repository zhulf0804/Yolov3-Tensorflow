
# for model
_batch_size = 6
_input_size = 416
_num_classes = 20
_anchor_per_scale = 3
_strides = [8, 16, 32]
_max_boxes = 80
_anchor_txt = './data/anchors.txt'
_data_txt = './data'


# for train
_max_steps = 50000
_print_steps = 200
_saved_steps = 5000


# for learning rate
_initial_lr = 1e-4
_end_lr = 1e-6
_decay_steps = 40000
_weight_decay = 1e-4
_power = 0.9


# for datasets
_num_train_samples = 5716
_num_val_samples = 5822
_voc_names = './utils/voc.names'


# for weights and summary
_coco_c_weights = './data/yolov3.weights'
_coco_tf_weights = './data/checkpoints'
_saved_weights = './checkpoints'
_saved_summary_train_path = './summary/train/'
_saved_summary_val_path = './summary/val/'


## for coco dataset
_coco_classes = 80
_coco_names = './utils/coco.names'