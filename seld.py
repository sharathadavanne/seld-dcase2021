#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
from metrics import SELD_evaluation_metrics
import keras_model
import parameter
import time
plot.switch_backend('agg')
from IPython import embed

def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = np.sqrt(x**2 + y**2 + z**2) > 0.5
      
    return sed, accdoa_in

def collect_test_labels(_data_gen_test, _data_out, _nb_classes, quick_test, is_accdoa, doa_objective):
    # Collecting ground truth for test data
    nb_batch = 2 if quick_test else _data_gen_test.get_total_batches_in_data()

    if is_accdoa:         
        batch_size = _data_out[0]
        gt_sed = np.zeros((nb_batch * batch_size, _data_out[1], _data_out[2]//3))
        gt_doa = np.zeros((nb_batch * batch_size, _data_out[1], _data_out[2]))

        print("nb_batch in test: {}".format(nb_batch))
        cnt = 0
        for tmp_feat, tmp_label in _data_gen_test.generate():            
            sed, doa = get_accdoa_labels(tmp_label, _nb_classes)
            gt_sed[cnt * batch_size:(cnt + 1) * batch_size, :, :] = sed
            gt_doa[cnt * batch_size:(cnt + 1) * batch_size, :, :] = doa
            cnt = cnt + 1
            if cnt == nb_batch:
                break
    else:
        batch_size = _data_out[0][0]
        gt_sed = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[0][2]))
        gt_doa = np.zeros((nb_batch * batch_size, _data_out[0][1], _data_out[1][2]))

        print("nb_batch in test: {}".format(nb_batch))
        cnt = 0
        for tmp_feat, tmp_label in _data_gen_test.generate():            
            gt_sed[cnt * batch_size:(cnt + 1) * batch_size, :, :] = tmp_label[0]
            if doa_objective is 'mse':
                doa_label = tmp_label[1]
            else:
                doa_label = tmp_label[1][:, :, _nb_classes:]
            gt_doa[cnt * batch_size:(cnt + 1) * batch_size, :, :] = doa_label
            cnt = cnt + 1
            if cnt == nb_batch:
                break
    return gt_sed.astype(int), gt_doa

 
def plot_functions(fig_name, _tr_loss, _new_metric, _new_seld_metric):
    plot.figure()
    nb_epoch = len(_tr_loss)
    plot.subplot(211)
    plot.plot(range(nb_epoch), _tr_loss, label='train loss')
    plot.legend()
    plot.grid(True)

    plot.subplot(212)
    plot.plot(range(nb_epoch), _new_metric[:, 0], label='seld er')
    plot.plot(range(nb_epoch), _new_metric[:, 1], label='seld f1')
    plot.plot(range(nb_epoch), _new_metric[:, 2]/180., label='doa er / 180')
    plot.plot(range(nb_epoch), _new_metric[:, 3], label='doa fr')
    plot.plot(range(nb_epoch), _new_seld_metric, label='seld')
    plot.legend()
    plot.grid(True)
    plot.close()


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.
    
    :param argv: expects two optional inputs. 
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """
    print(argv)
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameter.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    feat_cls = cls_feature_class.FeatureClass(params)
    train_splits, val_splits, test_splits = None, None, None

    if params['mode'] == 'dev':
        test_splits = [6]
        val_splits = [5]
        train_splits = [[1, 2, 3, 4]]

    elif params['mode'] == 'eval':
        test_splits = [[7, 8]]
        val_splits = [[6]]
        train_splits = [[1, 2, 3, 4, 5]]

    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        print('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        cls_feature_class.create_folder(params['model_dir'])
        unique_name = '{}_{}_{}_{}_split{}'.format(
            task_id, job_id, params['dataset'], params['mode'], split
        )
        unique_name = os.path.join(params['model_dir'], unique_name)
        model_name = '{}_model.h5'.format(unique_name)
        print("unique_name: {}\n".format(unique_name))

        # Load train and validation data
        print('Loading training dataset:')
        data_gen_train = cls_data_generator.DataGenerator(
            params=params, split=train_splits[split_cnt]
        )

        print('Loading validation dataset:')
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False
        )

        # Collect the reference labels for validation data
        data_in, data_out = data_gen_train.get_data_sizes()
        print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))

        nb_classes = data_gen_train.get_nb_classes()
        gt = collect_test_labels(data_gen_val, data_out, nb_classes, params['quick_test'], params['is_accdoa'], params['doa_objective'])
        sed_gt = SELD_evaluation_metrics.reshape_3Dto2D(gt[0])
        doa_gt = SELD_evaluation_metrics.reshape_3Dto2D(gt[1])

        print('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n\trnn_size: {}, fnn_size: {}\n\tdoa_objective: {}\n'.format(
            params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'],
            params['fnn_size'], params['doa_objective']))

        print('Using loss weights : {}'.format(params['loss_weights']))
        model = keras_model.get_model(data_in=data_in, data_out=data_out, dropout_rate=params['dropout_rate'],
                                      nb_cnn2d_filt=params['nb_cnn2d_filt'], f_pool_size=params['f_pool_size'], t_pool_size=params['t_pool_size'],
                                      rnn_size=params['rnn_size'], fnn_size=params['fnn_size'],
                                      weights=params['loss_weights'], doa_objective=params['doa_objective'], is_accdoa=params['is_accdoa'])
        best_seld_metric = 99999
        best_epoch = -1
        patience_cnt = 0
        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        tr_loss = np.zeros(nb_epoch)
        seld_metric = np.zeros((nb_epoch, 4))
        early_stop_metric = np.zeros(nb_epoch)

        # start training
        for epoch_cnt in range(nb_epoch):
            start = time.time()

            # train once per epoch
            hist = model.fit_generator(
                generator=data_gen_train.generate(),
                steps_per_epoch=2 if params['quick_test'] else data_gen_train.get_total_batches_in_data(),
                epochs=params['epochs_per_fit'],
                verbose=2,
            )
            tr_loss[epoch_cnt] = hist.history.get('loss')[-1]

            # predict once per epoch
            pred = model.predict_generator(
                generator=data_gen_val.generate(),
                steps=2 if params['quick_test'] else data_gen_val.get_total_batches_in_data(),
                verbose=2
            )


            if params['is_accdoa']:
                sed_pred, doa_pred = get_accdoa_labels(pred, nb_classes)
                sed_pred = SELD_evaluation_metrics.reshape_3Dto2D(sed_pred)
                doa_pred = SELD_evaluation_metrics.reshape_3Dto2D(doa_pred)
            else:
                sed_pred = SELD_evaluation_metrics.reshape_3Dto2D(pred[0]) > 0.5
                doa_pred = SELD_evaluation_metrics.reshape_3Dto2D(pred[1] if params['doa_objective'] is 'mse' else pred[1][:, :, nb_classes:])
            
            # Calculate the DCASE 2021 metrics - Location-aware detection and Class-aware localization scores
            cls_metric = SELD_evaluation_metrics.SELDMetrics(nb_classes=data_gen_val.get_nb_classes(), doa_threshold=params['lad_doa_thresh'])
            pred_dict = feat_cls.regression_label_format_to_output_format(
                sed_pred, doa_pred
            )
            gt_dict = feat_cls.regression_label_format_to_output_format(
                sed_gt, doa_gt
            )

            pred_blocks_dict = feat_cls.segment_labels(pred_dict, sed_pred.shape[0])
            gt_blocks_dict = feat_cls.segment_labels(gt_dict, sed_gt.shape[0])

            cls_metric.update_seld_scores_xyz(pred_blocks_dict, gt_blocks_dict)
            seld_metric[epoch_cnt, :] = cls_metric.compute_seld_scores()
            early_stop_metric[epoch_cnt] = SELD_evaluation_metrics.early_stopping_metric(seld_metric[epoch_cnt, :2], seld_metric[epoch_cnt, 2:])

            # Visualize the metrics with respect to epochs
            plot_functions(unique_name, tr_loss, seld_metric, early_stop_metric)

            patience_cnt += 1
            if early_stop_metric[epoch_cnt] < best_seld_metric:
                best_seld_metric = early_stop_metric[epoch_cnt]
                best_epoch = epoch_cnt
                model.save(model_name)
                patience_cnt = 0

            print(
                'epoch_cnt: {}, time: {:0.2f}s, tr_loss: {:0.2f}, '
                '\n\t\t DCASE2021 SCORES: ER: {:0.2f}, F: {:0.1f}, DE: {:0.1f}, DE_F:{:0.1f}, seld_score (early stopping score): {:0.2f}, '
                'best_seld_score: {:0.2f}, best_epoch : {}\n'.format(
                    epoch_cnt, time.time() - start, tr_loss[epoch_cnt],
                    seld_metric[epoch_cnt, 0], seld_metric[epoch_cnt, 1]*100,
                    seld_metric[epoch_cnt, 2], seld_metric[epoch_cnt, 3]*100,
                    early_stop_metric[epoch_cnt], best_seld_metric, best_epoch
                )
            )
            if patience_cnt > params['patience']:
                break

        print('\nResults on validation split:')
        print('\tUnique_name: {} '.format(unique_name))
        print('\tSaved model for the best_epoch: {}'.format(best_epoch))
        print('\tSELD_score (early stopping score) : {}'.format(best_seld_metric))

        print('\n\tDCASE2021 scores')
        print('\tClass-aware localization scores: DOA_error: {:0.1f}, F-score: {:0.1f}'.format(seld_metric[best_epoch, 2], seld_metric[best_epoch, 3]*100))
        print('\tLocation-aware detection scores: Error rate: {:0.2f}, F-score: {:0.1f}'.format(seld_metric[best_epoch, 0], seld_metric[best_epoch, 1]*100))

        # ------------------  Calculate metric scores for unseen test split ---------------------------------
        print('\nLoading the best model and predicting results on the testing split')
        print('\tLoading testing dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=split, shuffle=False, per_file=params['dcase_output'], is_eval=True if params['mode'] is 'eval' else False
        )

        model = keras_model.load_seld_model('{}_model.h5'.format(unique_name), params['doa_objective'])
        pred_test = model.predict_generator(
            generator=data_gen_test.generate(),
            steps=2 if params['quick_test'] else data_gen_test.get_total_batches_in_data(),
            verbose=2
        )
        if params['is_accdoa']:
            test_sed_pred, test_doa_pred = get_accdoa_labels(pred_test, nb_classes)
            test_sed_pred = SELD_evaluation_metrics.reshape_3Dto2D(test_sed_pred)
            test_doa_pred = SELD_evaluation_metrics.reshape_3Dto2D(test_doa_pred)
        else:
            test_sed_pred = SELD_evaluation_metrics.reshape_3Dto2D(pred_test[0]) > 0.5
            test_doa_pred = SELD_evaluation_metrics.reshape_3Dto2D(pred_test[1] if params['doa_objective'] is 'mse' else pred_test[1][:, :, nb_classes:])

        if params['dcase_output']:
            # Dump results in DCASE output format for calculating final scores
            dcase_dump_folder = os.path.join(params['dcase_dir'], '{}_{}_{}'.format(task_id, params['dataset'], params['mode']))
            cls_feature_class.create_folder(dcase_dump_folder)
            print('Dumping recording-wise results in: {}'.format(dcase_dump_folder))

            test_filelist = data_gen_test.get_filelist()
            # Number of frames for a 60 second audio with 100ms hop length = 600 frames
            max_frames_with_content = data_gen_test.get_nb_frames()

            # Number of frames in one batch (batch_size* sequence_length) consists of all the 600 frames above with
            # zero padding in the remaining frames
            frames_per_file = data_gen_test.get_frame_per_file()
            for file_cnt in range(test_sed_pred.shape[0]//frames_per_file):
                output_file = os.path.join(dcase_dump_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
                dc = file_cnt * frames_per_file
                output_dict = feat_cls.regression_label_format_to_output_format(
                    test_sed_pred[dc:dc + max_frames_with_content, :],
                    test_doa_pred[dc:dc + max_frames_with_content, :]
                )
                data_gen_test.write_output_format_file(output_file, output_dict)

        if params['mode'] is 'dev':
            test_data_in, test_data_out = data_gen_test.get_data_sizes()
            test_gt = collect_test_labels(data_gen_test, test_data_out, nb_classes, params['quick_test'], params['is_accdoa'], params['doa_objective'])
            test_sed_gt = SELD_evaluation_metrics.reshape_3Dto2D(test_gt[0])
            test_doa_gt = SELD_evaluation_metrics.reshape_3Dto2D(test_gt[1])
         
            # Calculate DCASE2021 scores
            cls_test_metric = SELD_evaluation_metrics.SELDMetrics(nb_classes=data_gen_test.get_nb_classes(), doa_threshold=params['lad_doa_thresh'])
            test_pred_dict = feat_cls.regression_label_format_to_output_format(
                test_sed_pred, test_doa_pred
            )
            test_gt_dict = feat_cls.regression_label_format_to_output_format(
                test_sed_gt, test_doa_gt
            )

            test_pred_blocks_dict = feat_cls.segment_labels(test_pred_dict, test_sed_pred.shape[0])
            test_gt_blocks_dict = feat_cls.segment_labels(test_gt_dict, test_sed_gt.shape[0])

            cls_test_metric.update_seld_scores_xyz(test_pred_blocks_dict, test_gt_blocks_dict)
            test_seld_metric = cls_test_metric.compute_seld_scores()
            test_early_stop_metric = SELD_evaluation_metrics.early_stopping_metric(test_seld_metric[:2], test_seld_metric[2:])

            print('Results on test split:')

            print('\tDCASE2021 Scores')
            print('\tClass-aware localization scores: DOA Error: {:0.1f}, F-score: {:0.1f}'.format(test_seld_metric[2], test_seld_metric[3]*100))
            print('\tLocation-aware detection scores: Error rate: {:0.2f}, F-score: {:0.1f}'.format(test_seld_metric[0], test_seld_metric[1]*100))
            print('\tSELD (early stopping metric): {:0.2f}'.format(test_early_stop_metric))



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)
