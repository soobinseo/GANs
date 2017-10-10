filter_width = 5
filter_height = 5
stride = 1

im_height = 64
im_width = 64
num_channel = 3

encoder_hp = [4*64,64,128]
decoder_hp = [[8,8,64],[16,16,32],[32,32,16]]

disc_filter_height = 5
disc_filter_width = 5
disc_hp = [64,128,256]

keep_prob = 0.5
loss_fn = "standard" # ['standard','ls','w','impw','dra']

num_epochs = 100
batch_size = 32
save_dir = './log_dir_' + loss_fn
