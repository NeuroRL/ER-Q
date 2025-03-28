from tqdm import tqdm


def simulation(num_sims, num_epis, agent, env, collector):
    for sim in range(num_sims):
        agent.initialize()
        collector.initialize()
        env.initialize()

        print(f"[{collector._taskname} {sim+1}/{num_sims}]", flush=True)
        with tqdm(range(num_epis),
                  bar_format="{percentage:3.0f}% | {bar} | {n_fmt}/{total_fmt}, {elapsed}/{remaining}, {rate_fmt}{postfix}",
                  desc=f"[{collector._taskname} {sim+1}/{num_sims}]",
                  dynamic_ncols=True) as pbar:

            for epi in pbar:
                postfix = f"[ret: {collector.return_nsim_mean(n=20):1.2f}/{collector.return_nsim_mean(n=epi):1.2f}], [step: {collector.steps_nsim_mean(n=20):2.1f}/{collector.steps_nsim_mean(n=epi):2.1f}]"
                pbar.set_postfix_str(postfix)
                observation = env.reset()
                agent.reset(observation)
                collector.reset(epi)

                done = False
                while not done:
                    act = agent.act()
                    agent.unbiased_policy_distribution
                    agent.biased_policy_distribution
                    observation, reward, done, _ = env.step(act)
                    agent.observe(observation=observation, reward=reward, done=done, epis=epi)
                    collector.collect_step_data()
                collector.collect_episodic_data()
        collector.collect_simulation_data()
        collector.save()
