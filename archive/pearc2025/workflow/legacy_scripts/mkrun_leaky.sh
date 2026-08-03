python tfrun.py summary_metadata.csv config_scan.json '{"activations":["leaky_relu"],"losses":["mse"],"epochs":1000}' 2>&1 > runs/leaky_relu_mse.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["leaky_relu"],"losses":["mae"],"epochs":1000}' 2>&1 > runs/leaky_relu_mae.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["leaky_relu"],"losses":["kl_divergence"],"epochs":1000}' 2>&1 > runs/leaky_relu_kl_divergence.out
