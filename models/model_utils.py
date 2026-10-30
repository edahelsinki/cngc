from models.linear import LinearRegressor
from models.mlp import MLPRegressor
from models.cnn import CNN1dRegressor
from models.transformer import TransfEncoder


def get_model(model_name, x_shape, y_shape, dataset_name, hidden_size=20):
    in_features = x_shape[1]
    out_features = y_shape[1]
    if model_name == 'linear':
        model = LinearRegressor(in_features, out_features, add_bias=False)
    elif model_name == 'mlp':
        model = MLPRegressor(in_features, out_features, hidden_layers=[hidden_size, hidden_size], add_bias=False, is_relu=True)
    elif model_name == 'cnn':
        if dataset_name == 'occupancy' or dataset_name == 'air'or dataset_name.startswith('gen_seq'):
            seq_len = x_shape[-1]
            model = CNN1dRegressor(in_channels=in_features, out_channels=hidden_size, kernel_size=5, n_layers=2, seq_len=seq_len, pool_size=2)
        else:
            raise ValueError(f"CNN model is not implemented for dataset {dataset_name}")
    elif model_name == 'transf':
        if dataset_name == 'occupancy' or dataset_name == 'air' or dataset_name.startswith('gen_seq'):
            model = TransfEncoder(n_blocks=2, feat=in_features, d_model=8, n_out=1, dim_ff=8, nhead=2, p=0.05)
        else:
            raise ValueError(f"Transformer model is not implemented for dataset {dataset_name}")
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model
