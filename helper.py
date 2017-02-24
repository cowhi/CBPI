import os
import sys
import shutil
import errno
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import scipy as sp
import scipy.stats
import csv
import logging
sns.set(style="darkgrid")


def create_dir(path_to_dir):
    _logger = logging.getLogger(__name__)
    if not os.path.isdir(path_to_dir):
        try:
            os.makedirs(path_to_dir)
        except:
            _logger.critical("Can't create directory - %s" % str(path_to_dir))
            sys.exit()
    return path_to_dir


def copy_file(src, dest):
    _logger = logging.getLogger(__name__)
    try:
        shutil.copy(src, dest)
    # eg. src and dest are the same file
    except shutil.Error as e:
        _logger.critical("Can't copy file - %s" % str(e))
        sys.exit()
    # eg. source or destination doesn't exist
    except IOError as e:
        _logger.critical("Can't copy file - %s" % str(e))
        sys.exit()


def write_stats_file(path_to_file, *args):
    _logger = logging.getLogger(__name__)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    # creating new line for file
    line = ''
    for arg in args:
        if type(arg) is list:
            for elem in arg:
                line += str(elem) + ','
        else:
            line += str(arg) + ','
    line = line[:-1] + '\n'
    # write to file
    try:
        file_handle = os.open(path_to_file, flags)
    except OSError as e:
        if e.errno == errno.EEXIST:  # Failed as the file already exists.
            with open(path_to_file, 'a+') as f:
                f.write(line)
        else:  # Something unexpected went wrong so reraise the exception.
            _logger.critical("Can't write stats file - %s " % str(e))
            sys.exit()
    else:  # No exception, so the file must have been created successfully.
        with os.fdopen(file_handle, 'w') as file_obj:
            # Using `os.fdopen` converts the handle to an object that acts
            # like a regular Python file object, and the `with` context
            # manager means the file will be automatically closed when
            # we're done with it.
            file_obj.write(line)


def mean_confidence_interval(my_list, confidence=0.95):
    my_array = 1.0 * np.array(my_list)
    array_mean, array_se = np.mean(my_array), scipy.stats.sem(my_array)
    margin = array_se * sp.stats.t._ppf((1 + confidence) / 2.,
                                        len(my_array) - 1)
    return array_mean, array_mean - margin, array_mean + margin


def summarize_runs(path_to_dir):
    _logger = logging.getLogger(__name__)
    run_dirs = glob.glob(os.path.join(path_to_dir) + '/*/')
    run_files = [os.path.join(run_dir, 'stats_run.csv')
                 for run_dir in run_dirs]
    df = pd.concat((pd.read_csv(run_file) for run_file in run_files))
    steps = df.groupby(['episode'])['steps_mean']
    steps = list(steps)
    reward = df.groupby(['episode'])['reward_mean']
    reward = list(reward)
    summary = []
    for episode in range(0, len(reward)):
        step_mean, step_lower, step_upper = \
            mean_confidence_interval(steps[episode][1])
        reward_mean, reward_lower, reward_upper = \
            mean_confidence_interval(reward[episode][1])
        summary.append([int(steps[episode][0]), step_mean,
                        step_lower, step_upper,
                        reward_mean, reward_lower, reward_upper])
    header = ['episode', 'steps_mean', 'steps_lower', 'steps_upper',
              'reward_mean', 'reward_lower', 'reward_upper']
    try:
        with open(os.path.join(path_to_dir, 'stats_task.csv'), 'w') \
                as csvfile:
            writer = csv.writer(csvfile,
                                dialect='excel',
                                quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow(header)
            for data in summary:
                writer.writerow(data)
    except IOError as e:
        _logger.critical("Can't write stats file - %s " % str(e))
        sys.exit()


def plot_run(path_to_dir):
    df = pd.read_csv(os.path.join(path_to_dir, 'stats_run.csv'))
    # print(df)
    for column in df.columns:
        plt.figure(figsize=(10, 6), dpi=80)
        plt.plot(df['episode'], df[column],
                 label=column, color='blue', linewidth=2.0)
        plt.ylabel(column, fontsize=20, fontweight='bold')
        plt.xlabel('episodes', fontsize=20, fontweight='bold')
        plt.legend()
        plt.savefig(os.path.join(path_to_dir, 'plot_' + str(column) + '.png'))
        plt.close('all')


def plot_runs(path_to_dir):
    run_dirs = glob.glob(os.path.join(path_to_dir) + '/*/')
    dfs = []
    for run_dir in run_dirs:
        dfs.append(pd.read_csv(os.path.join(run_dir, 'stats_run.csv')))
    for column in dfs[0].columns:
        plt.figure(figsize=(10, 6), dpi=80)
        run_count = 1
        for df in dfs:
            plt.plot(df['episode'], df[column],
                     label=column+'_'+str(run_count), linewidth=2.0)
            run_count += 1
        plt.ylabel(column, fontsize=20, fontweight='bold')
        plt.xlabel('episodes', fontsize=20, fontweight='bold')
        plt.legend()
        plt.savefig(os.path.join(path_to_dir, 'plot_' + str(column) + '.png'))
        plt.close('all')


def plot_task(path_to_dir):
    df = pd.read_csv(os.path.join(path_to_dir, 'stats_task.csv'))
    factors = ['steps', 'reward']
    colors = ['blue', 'green']
    for factor, color in zip(factors, colors):
        plt.figure(figsize=(10, 6), dpi=80)
        plt.plot(df['episode'], df[factor + '_mean'],
                 label=factor+'_mean', color=color, linewidth=2.0)
        plt.plot(df['episode'], df[factor + '_lower'],
                 label=factor+'_lower', color=color, linewidth=1.0)
        plt.plot(df['episode'], df[factor + '_upper'],
                 label=factor+'_upper', color=color, linewidth=1.0)
        plt.fill_between(df['episode'], df[factor + '_mean'],
                         df[factor + '_lower'],
                         facecolor=color, alpha=0.2)
        plt.fill_between(df['episode'], df[factor + '_mean'],
                         df[factor + '_upper'],
                         facecolor=color, alpha=0.2)
        plt.ylabel(factor, fontsize=20, fontweight='bold')
        plt.xlabel('episodes', fontsize=20, fontweight='bold')
        plt.legend(fontsize=14)
        plt.savefig(os.path.join(path_to_dir, 'plot_' + str(factor) + '.png'))
        plt.close('all')


def plot_stats_libs(path_to_dir):
    df = pd.read_csv(os.path.join(path_to_dir, 'stats_libs.csv'))
    plt.figure(figsize=(10, 6), dpi=80)
    df.plot(x='episode')
    plt.ylabel('policy probability [%]', fontsize=20, fontweight='bold')
    plt.xlabel('episodes', fontsize=20, fontweight='bold')
    plt.legend(fontsize=14)
    plt.savefig(os.path.join(path_to_dir, 'plot_stats_libs.png'))
    plt.close('all')
