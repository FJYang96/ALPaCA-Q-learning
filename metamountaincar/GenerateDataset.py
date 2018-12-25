from MCDataset import MCOnlineSampleDataset
from VIAgent import MCVI
import yaml

# Parameters
ITERATIONS = 300
N_FUNCS = 10
N_SAMPLES = 80
close_dir = './MC_Q_samples/close/'
mid_dir = './MC_Q_samples/mid/'
far_dir = './MC_Q_samples/far/'

# Pretty close to the original dynamics
config = {}
config['gran'] = 150
config['step_limit'] = 200
config['pos_range'] = [-1.2, 0.6]
config['vel_range'] = [-0.07, 0.07]
config['thrust_range'] = [0.0009, 0.0011]
config['gravity_range'] = [0.0024, 0.0027]
config['sigma_eps'] = 0.01
config['x_dim'] = 2
config['y_dim'] = 3

for i in range(ITERATIONS):
    online_data = MCOnlineSampleDataset(config)
    xy = online_data.sample(N_FUNCS, N_SAMPLES)
    xy_dump = yaml.dump(xy)
    with open(close_dir+str(i)+'.yml', 'w') as f:
        f.write(xy_dump)
    print("Finished sample", i)

# Further to the original dynamics
config['thrust_range'] = [0.00075, 0.00125]
config['gravity_range'] = [0.0022, 0.0028]
for i in range(ITERATIONS):
    online_data = MCOnlineSampleDataset(config)
    xy = online_data.sample(N_FUNCS, N_SAMPLES)
    xy_dump = yaml.dump(xy)
    with open(mid_dir+str(i)+'.yml', 'w') as f:
        f.write(xy_dump)
    print("Finished sample", i)

# Furthest from original dynamics
config['thrust_range'] = [0.005, 0.0015]
config['gravity_range'] = [0.002, 0.003]
for i in range(ITERATIONS):
    online_data = MCOnlineSampleDataset(config)
    xy = online_data.sample(N_FUNCS, N_SAMPLES)
    xy_dump = yaml.dump(xy)
    with open(far_dir+str(i)+'.yml', 'w') as f:
        f.write(xy_dump)
    print("Finished sample", i)
