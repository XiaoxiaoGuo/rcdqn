import agents.RCAgent as RCAgent


def get_agent(config, env, log_dir):
    if config.algo == 'rcdqn':
        return RCAgent.ReadingComprehensionDQN(
            args=config, env=env, log_dir=log_dir)
    else:
        print('unknown algorithm: {}'.format(config.algo))
        exit(0)
    return
