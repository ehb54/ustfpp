mkrun_python tfrun.py summary_metadata.csv config_scan.json '{"activations":["relu"],"losses":["mse"],"epochs":1000}' 2>&1 > run_relu_mse.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["relu"],"losses":["mae"],"epochs":1000}' 2>&1 > run_relu_mae.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["relu"],"losses":["kl_divergence"],"epochs":1000}' 2>&1 > run_relu_kl_divergence.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["sigmoid"],"losses":["mse"],"epochs":1000}' 2>&1 > run_sigmoid_mse.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["sigmoid"],"losses":["mae"],"epochs":1000}' 2>&1 > run_sigmoid_mae.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["sigmoid"],"losses":["kl_divergence"],"epochs":1000}' 2>&1 > run_sigmoid_kl_divergence.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["softmax"],"losses":["mse"],"epochs":1000}' 2>&1 > run_softmax_mse.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["softmax"],"losses":["mae"],"epochs":1000}' 2>&1 > run_softmax_mae.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["softmax"],"losses":["kl_divergence"],"epochs":1000}' 2>&1 > run_softmax_kl_divergence.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["softplus"],"losses":["mse"],"epochs":1000}' 2>&1 > run_softplus_mse.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["softplus"],"losses":["mae"],"epochs":1000}' 2>&1 > run_softplus_mae.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["softplus"],"losses":["kl_divergence"],"epochs":1000}' 2>&1 > run_softplus_kl_divergence.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["softsign"],"losses":["mse"],"epochs":1000}' 2>&1 > run_softsign_mse.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["softsign"],"losses":["mae"],"epochs":1000}' 2>&1 > run_softsign_mae.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["softsign"],"losses":["kl_divergence"],"epochs":1000}' 2>&1 > run_softsign_kl_divergence.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["tanh"],"losses":["mse"],"epochs":1000}' 2>&1 > run_tanh_mse.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["tanh"],"losses":["mae"],"epochs":1000}' 2>&1 > run_tanh_mae.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["tanh"],"losses":["kl_divergence"],"epochs":1000}' 2>&1 > run_tanh_kl_divergence.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["selu"],"losses":["mse"],"epochs":1000}' 2>&1 > run_selu_mse.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["selu"],"losses":["mae"],"epochs":1000}' 2>&1 > run_selu_mae.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["selu"],"losses":["kl_divergence"],"epochs":1000}' 2>&1 > run_selu_kl_divergence.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["elu"],"losses":["mse"],"epochs":1000}' 2>&1 > run_elu_mse.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["elu"],"losses":["mae"],"epochs":1000}' 2>&1 > run_elu_mae.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["elu"],"losses":["kl_divergence"],"epochs":1000}' 2>&1 > run_elu_kl_divergence.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["exponential"],"losses":["mse"],"epochs":1000}' 2>&1 > run_exponential_mse.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["exponential"],"losses":["mae"],"epochs":1000}' 2>&1 > run_exponential_mae.out
python tfrun.py summary_metadata.csv config_scan.json '{"activations":["exponential"],"losses":["kl_divergence"],"epochs":1000}' 2>&1 > run_exponential_kl_divergence.out
.sh
