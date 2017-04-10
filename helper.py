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


def delete_dirs(path_to_dir):
    _logger = logging.getLogger(__name__)
    episode_dirs = glob.glob(os.path.join(path_to_dir) + '/episode_*/')
    try:
        for episode_dir in episode_dirs:
            shutil.rmtree(episode_dir)
    except:
        _logger.critical("Can't delete directory - %s" % str(episode_dir))
        sys.exit()


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


def summarize_runs_results(path_to_dir):
    _logger = logging.getLogger(__name__)
    run_dirs = glob.glob(os.path.join(path_to_dir) + '/*/')
    run_files = [os.path.join(run_dir, 'stats_run.csv')
                 for run_dir in run_dirs]
    df = pd.concat((pd.read_csv(run_file) for run_file in run_files))
    steps = df.groupby(['episode'])['steps_mean']
    steps = list(steps)
    reward = df.groupby(['episode'])['reward_mean']
    reward = list(reward)
    effort = df.groupby(['episode'])['step_count']
    effort = list(effort)
    summary = []
    for episode in range(0, len(reward)):
        step_mean, step_lower, step_upper = \
            mean_confidence_interval(steps[episode][1])
        reward_mean, reward_lower, reward_upper = \
            mean_confidence_interval(reward[episode][1])
        effort_mean, effort_lower, effort_upper = \
            mean_confidence_interval(effort[episode][1])
        summary.append([int(steps[episode][0]),
                        step_mean, step_lower, step_upper,
                        reward_mean, reward_lower, reward_upper,
                        effort_mean, effort_lower, effort_upper])
    header = ['episode', 'steps_mean', 'steps_lower', 'steps_upper',
              'reward_mean', 'reward_lower', 'reward_upper',
              'effort_mean', 'effort_lower', 'effort_upper']
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


def summarize_runs_policy_choice(path_to_dir, kind='probs'):
    _logger = logging.getLogger(__name__)
    run_dirs = glob.glob(os.path.join(path_to_dir) + '/*/')
    policy_files = [os.path.join(run_dir, 'stats_policy_' + kind + '.csv')
                    for run_dir in run_dirs]
    df = pd.concat((pd.read_csv(policy_usage_file)
                    for policy_usage_file in policy_files))
    policies = list(df)
    policies = [policy for policy in policies if 'episode' not in policy]
    for policy in policies:
        usage = df.groupby(['episode'])[policy]
        usage = list(usage)
        summary = []
        for episode in range(0, len(usage)):
            mean_value, lower_value, upper_value = \
               mean_confidence_interval(usage[episode][1])
            summary.append([int(usage[episode][0]),
                            mean_value, lower_value, upper_value])
        header = ['episode', 'mean', 'lower', 'upper']
        try:
            with open(os.path.join(path_to_dir,
                                   kind + '_'+str(policy)+'.csv'),
                      'w') as csvfile:
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
        plt.figure(figsize=(10, 4), dpi=80)
        plt.plot(df['episode'], df[column],
                 label=column, color='blue', linewidth=2.0)
        plt.ylabel(column, fontsize=20, fontweight='bold')
        plt.xlabel('episodes', fontsize=20, fontweight='bold')
        plt.legend()
        plt.savefig(os.path.join(path_to_dir, 'plot_' + str(column) + '.png'),
                    bbox_inches='tight')
        plt.close('all')


def plot_runs(path_to_dir):
    run_dirs = glob.glob(os.path.join(path_to_dir) + '/*/')
    dfs = []
    for run_dir in run_dirs:
        dfs.append(pd.read_csv(os.path.join(run_dir, 'stats_run.csv')))
    for column in dfs[0].columns:
        plt.figure(figsize=(10, 4), dpi=80)
        run_count = 1
        for df in dfs:
            plt.plot(df['episode'], df[column],
                     label=column+'_'+str(run_count), linewidth=2.0)
            run_count += 1
        plt.ylabel(column, fontsize=20, fontweight='bold')
        plt.xlabel('episodes', fontsize=20, fontweight='bold')
        plt.legend()
        plt.savefig(os.path.join(path_to_dir, 'plot_' + str(column) + '.png'),
                    bbox_inches='tight')
        plt.close('all')


def plot_task(path_to_dir):
    df = pd.read_csv(os.path.join(path_to_dir, 'stats_task.csv'))
    factors = ['steps', 'reward', 'effort']
    colors = ['blue', 'green', 'red']
    for factor, color in zip(factors, colors):
        plt.figure(figsize=(10, 4), dpi=80)
        if factor == 'steps':
            df[factor + '_mean'] = df[factor + '_mean'].clip(0.0, 100.0)
            df[factor + '_lower'] = df[factor + '_lower'].clip(0.0, 100.0)
            df[factor + '_upper'] = df[factor + '_upper'].clip(0.0, 100.0)
        if factor == 'reward':
            df[factor + '_mean'] = df[factor + '_mean'].clip(0.0, 1.0)
            df[factor + '_lower'] = df[factor + '_lower'].clip(0.0, 1.0)
            df[factor + '_upper'] = df[factor + '_upper'].clip(0.0, 1.0)
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
        plt.savefig(os.path.join(path_to_dir, 'plot_' + str(factor) + '.png'),
                    bbox_inches='tight')
        plt.close('all')


def plot_policy_choice(path_to_dir, kind='probs'):
    if kind == 'probs':
        ylabel = 'policy probability [%]'
    elif kind == 'absolute':
        ylabel = 'policy mean [steps]'
    elif kind == 'W':
        ylabel = 'Reuse gain [gain per episode]'
    elif kind == 'U':
        ylabel = 'policy usage [count]'
    elif kind == 'P':
        ylabel = 'policy probability [%]'
    else:
        pass
    df = pd.read_csv(os.path.join(path_to_dir,
                                  'stats_policy_' + kind + '.csv'))
    plt.figure(figsize=(10, 4), dpi=80)
    df.plot(x='episode')
    plt.ylabel(ylabel, fontsize=20, fontweight='bold')
    plt.xlabel('episodes', fontsize=20, fontweight='bold')
    plt.legend(fontsize=14)
    plt.savefig(os.path.join(path_to_dir, 'plot_policy_' + kind + '.png'),
                bbox_inches='tight')
    plt.close('all')


def plot_policy_choice_summary(path_to_dir, kind='probs'):
    limit_lower = 0
    if kind == 'probs':
        ylabel = 'policy probability [%]'
        skip = 6
        limit_upper = 1.0
    elif kind == 'absolute':
        ylabel = 'policy mean [steps]'
        skip = 9
        limit_upper = 100.0
    elif kind == 'W':
        ylabel = 'Reuse gain [gain per episode]'
        skip = 2
        limit_upper = 1.0
    elif kind == 'U':
        ylabel = 'policy usage [count]'
        skip = 2
        limit_upper = 1000.0
    elif kind == 'P':
        ylabel = 'policy probability [%]'
        skip = 2
        limit_upper = 1.0
    else:
        pass

    policy_files = glob.glob(
        os.path.join(path_to_dir) + '/' + kind + '_*.csv')
    colors = ['red', 'green', 'blue', 'yellow', 'black', 'brown', 'orange']
    plt.figure(figsize=(10, 4), dpi=80)
    color_count = 0
    for policy_file in policy_files:
        df = pd.read_csv(policy_file)
        policy_name = policy_file.split('/')
        policy_name = policy_name[-1].split('.')
        policy_name = policy_name[0][skip:]
        df['mean'] = df['mean'].clip(limit_lower, limit_upper)
        df['lower'] = df['lower'].clip(limit_lower, limit_upper)
        df['upper'] = df['upper'].clip(limit_lower, limit_upper)
        plt.plot(df['episode'], df['mean'],
                 label=policy_name, color=colors[color_count], linewidth=2.0)
        plt.plot(df['episode'], df['lower'],
                 label='_nolegend_', color=colors[color_count], linewidth=1.0)
        plt.plot(df['episode'], df['upper'],
                 label='_nolegend_', color=colors[color_count], linewidth=1.0)
        plt.fill_between(df['episode'], df['mean'],
                         df['lower'],
                         facecolor=colors[color_count], alpha=0.2)
        plt.fill_between(df['episode'], df['mean'],
                         df['upper'],
                         facecolor=colors[color_count], alpha=0.2)
        color_count += 1
    plt.ylabel(ylabel, fontsize=20, fontweight='bold')
    plt.xlabel('episodes', fontsize=20, fontweight='bold')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.xlim(0, 1000)
    plt.legend(fontsize=14, loc='upper left')
    plt.savefig(os.path.join(path_to_dir, 'plot_policy_' + kind + '.png'),
                bbox_inches='tight')
    plt.close('all')
