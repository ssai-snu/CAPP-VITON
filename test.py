import argparse
import os
import glob
import time
import yaml


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--step', type=str, default='all', help='step1, step2, step3, step4, all')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e. g. 0  0,1,2,  0,2. use -1 for CPU')

    opt = parser.parse_args()
    return opt


def get_config(config):
    opt = yaml.load(open(config.config), Loader=yaml.FullLoader)

    opt['name'] = config.name
    opt['gpu_ids'] = config.step
    opt['mode'] = config.mode
    opt['test_mode'] = config.test_mode

    return opt


def step1(opt):
    files = glob.glob(opt['background_step1_input_dir'] + '/*')
    file_names = ''
    for file in files:
        file_names += file + ' '
    os.system(opt['background_python'] + ' ' + os.path.join(opt['background_dir'], 'demo.py') +
              ' --config-file ' + os.path.join(opt['background_dir'], 'configs/CondInst/MS_R_101_3x.yaml') +
              ' --input ' + file_names +
              ' --output ' + opt['background_step1_output_dir'] +
              ' --step step1 ' +
              ' --opts MODEL.WEIGHTS ' + os.path.join(opt['background_dir'], 'checkpoint/CondInst_MS_R_101_3x.pth'))


def step2(opt):
    os.system(opt.background_python + ' ' + os.path.join(opt['openpose_dir'], 'test.py'))
    os.system(opt.background_python + ' ' + os.path.join(opt['dataset_dir'], 'gen_val_list.py'))


def step3(opt):
    # CIHP_PGN
    os.system(opt['cihp_python'] + ' ' + os.path.join(opt['cihp_pgn_dir'], 'test_pgn.py') +
              ' --data_dir ' + opt['cihp_input_dir'] +
              ' --list_path ' + opt['cihp_list_path'] +
              ' --data_id_list ' + opt['cihp_data_id_list'] +
              ' --checkpoint ' + os.path.join(opt['cihp_pgn_dir'], opt['cihp_checkpoint']) +
              ' --parsing_dir ' + opt['cihp_output_dir'] +
              ' --ref_image ' + opt['cihp_ref_img'])


def step4(opt):
    # VITON-HD
    os.system(opt['viton_hd_python'] + ' ' + os.path.join(opt['viton_hd_dir'], 'test.py') +
              ' --data_dir ' + opt['dataset_dir'] +
              ' --data_list ' + opt['viton_hd_dataset_list'] +
              ' --label_dir ' + opt['cihp_output_dir'] +
              ' --save_dir ' + opt['viton_hd_output_dir'] +
              ' --name ' + opt['name'] +
              ' --config ' + opt['config'] +
              ' --gpu_ids' + opt['gpu_ids'] +
              ' --which_epoch' + opt['which_epoch'] +
              ' --test_mode all')


def step5(opt):
    # background
    files = glob.glob(opt['viton_hd_output_dir'] + '/*')
    file_names = ''
    for file in files:
        file_names += file + ' '

    os.system(opt['background_python'] + ' ' + os.path.join(opt['background_dir'], 'demo.py') +
              ' --config-file ' + os.path.join(opt['background_dir'], 'configs/CondInst/MS_R_101_3x.yaml') +
              #  ' --config-file ' + os.path.join(opt['background_dir'], 'configs/BlendMask/R_101_3x.yaml') +
              ' --input ' + file_names +
              ' --bg ' + opt['background_info'] +
              ' --output ' + opt['background_final_output_dir'] +
              ' --step step3 ' +
              ' --opts MODEL.WEIGHTS ' + os.path.join(opt['background_dir'], 'checkpoint/CondInst_MS_R_101_3x.pth'))


def sec_to_hours(seconds):
    a = str(seconds // 3600)
    b = str((seconds % 3600) // 60)
    c = str((seconds % 3600) % 60)
    d = ["{} hours {} mins {} seconds".format(a, b, c)]
    return d


def main():
    config = get_opt()
    opt = get_config(config)

    opt['gpu_ids'] = config.gpu_ids

    start_t = time.time()

    if not os.path.exists(opt['background_step1_output_dir']):
        os.makedirs(opt['background_step1_output_dir'])
    if not os.path.exists(opt['image_dir']):
        os.makedirs(opt['image_dir'])
    if not os.path.exists(opt['pose_dir']):
        os.makedirs(opt['pose_dir'])
    if not os.path.exists(opt['pose_json_dir']):
        os.makedirs(opt['pose_json_dir'])
    if not os.path.exists(opt['cihp_output_dir']):
        os.makedirs(opt['cihp_output_dir'])
    if not os.path.exists(opt['viton_hd_output_dir']):
        os.makedirs(opt['viton_hd_output_dir'])
    if not os.path.exists(opt['background_final_output_dir']):
        os.makedirs(opt['background_final_output_dir'])

    if opt.step == 'step1':
        step1(opt)
        print("Step1 time : ", sec_to_hours(time.time() - start_t))

    elif opt.step == 'step2':
        step2(opt)
        print("Step2 time : ", sec_to_hours(time.time() - start_t))

    elif opt.step == 'step3':
        print("Step3 time : ", sec_to_hours(time.time() - start_t))

    elif opt.step == 'step4':
        print("Step4 time : ", sec_to_hours(time.time() - start_t))

    elif opt.step == 'step5':
        print("Step5 time : ", sec_to_hours(time.time() - start_t))

    elif opt.step == 'all':
        step1(opt)
        step1_t = time.time()
        step2(opt)
        step2_t = time.time()
        step3(opt)
        step3_t = time.time()
        step4(opt)
        step4_t = time.time()
        step5(opt)
        step5_t = time.time()
        print("Step1 time : ", sec_to_hours(step1_t - start_t))
        print("Step2 time : ", sec_to_hours(step2_t - step1_t))
        print("Step3 time : ", sec_to_hours(step3_t - step2_t))
        print("Step4 time : ", sec_to_hours(step4_t - step3_t))
        print("Step5 time : ", sec_to_hours(step5_t - step4_t))
        print("Total time : ", sec_to_hours(step4_t - start_t))


if __name__ == "__main__":
    main()
