def old_reward_plot(logdir, savedir=None):
    if savedir:
        savepath = Path(savedir)

    logpath = Path(logdir)

    config_file = next(logpath.glob('**/*.gin'))
    print(f'config file {config_file}')
    gin.parse_config_file(config_file, skip_unknown=True)

    bias = gin.config.query_parameter('Game.bias')
    num_points = gin.config.query_parameter('Game.num_points')
    circle_loss = CircleL1(num_points)

    run_logs = []
    for path in logpath.glob('**/*.json'):
        print(f'plotting from {path}')
        with open(path, 'r') as logfile:
            run_logs.append(pd.read_json(logfile))

    logs = pd.concat(run_logs, ignore_index=True)
    sender = pd.DataFrame(logs['sender'].to_list()).join(logs['epoch'])
    recver = pd.DataFrame(logs['recver'].to_list()).join(logs['epoch'])

    sns.set()

    # Rewards
    sns.lineplot(data=sender, x="epoch", y="reward", label="sender")
    sns.lineplot(data=recver, x="epoch", y="reward", label="recver")

    # Baselines
    nocomm_loss = torch.tensor(num_points / 4)
    nocomm_rew = - circle_loss(torch.tensor(0.), nocomm_loss)
#     oneshot_loss = env.bias_space.low / 2 + (env.bias_space.range) / 4
    fair_loss = bias / 2
    fair_rew = - circle_loss(torch.tensor(0.), fair_loss)
    plt.axhline(nocomm_rew, label='no communication', color="black", dashes=(2,2,2,2))
    plt.axhline(fair_rew, label='fair split', color="grey", dashes=(2,2,2,2))

    plt.legend()
    if savedir:
        plt.savefig(savepath / 'rewards.png')
    plt.show()
    plt.clf()

def old_plot(logdir, savedir):
    if savedir:
        savepath = Path(savedir)

    logpath = Path(logdir)

    config_file = next(logpath.glob('**/*.gin'))
    gin.parse_config_file(config_file, skip_unknown=True)
    env = ISR()

    run_logs = []
    for path in logpath.glob('**/*.json'):
        print(f'plotting from {path}')
        with open(path, 'r') as logfile:
            run_logs.append(pd.read_json(logfile))

    logs = pd.concat(run_logs, ignore_index=True)
    sender = pd.DataFrame(logs['sender'].to_list()).join(logs['episode'])
    recver = pd.DataFrame(logs['recver'].to_list()).join(logs['episode'])

    sns.set()

    # Rewards
    sns.lineplot(data=sender, x="episode", y="reward", label="sender")
    sns.lineplot(data=recver, x="episode", y="reward", label="recver")

    # Baselines
    nocomm_loss = torch.tensor(env.observation_space.n / 4)
    nocomm_rew = env._reward(nocomm_loss)
    oneshot_loss = env.bias_space.low / 2 + (env.bias_space.range) / 4
    oneshot_rew = env._reward(oneshot_loss)
    plt.axhline(nocomm_rew, label='no communication', color="black", dashes=(2,2,2,2))
    plt.axhline(oneshot_rew, label='fair split', color="grey", dashes=(2,2,2,2))

    plt.legend()
    if savedir:
        plt.savefig(savepath / 'rewards.png')
    plt.show()
    plt.clf()

def alternative_score(results_dir):
    # average of last 10 epochs
    results_path = Path(results_dir)

    run_logs = []
    for path in results_path.glob('*/logs.json'):
#         print(path)
        with open(path, 'r') as logfile:
            run_logs.append(pd.read_json(logfile))

    logs = pd.concat(run_logs, ignore_index=True)
    epoch = logs['epoch']
    sender = pd.DataFrame(logs['sender'].to_list()).join(logs['epoch'])
    recver = pd.DataFrame(logs['recver'].to_list()).join(logs['epoch'])
    total_error = sender['test_error'] + recver['test_error']
    return total_error.to_frame().groupby(epoch).mean()[-10:].mean()['test_error']

def get_alt_scores(results_dir):
    results_path = Path(results_dir)
    results = pd.read_csv(results_path / 'results.csv')
    alt_scores = []

    for bias, score, run_id in results.sort_values('bias').itertuples(index=False):
        logdir = list(results_path.glob(f'*{run_id}*'))[0]
        alt_score = alternative_score(logdir)
        alt_scores.append(alt_score)

    return results['bias'].sort_values().tolist(), alt_scores
