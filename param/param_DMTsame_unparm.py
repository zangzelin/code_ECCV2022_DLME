import argparse


def GetParamcoil20():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='coil20_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='coil20', )
    parser.add_argument('--data_trai_n', type=int, default=8000, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=False, )
    parser.add_argument('--perplexity', type=int, default=10, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, default=[-1, 600, 500, 400, 300, 200, 2], )

    parser.add_argument('--pow', type=float, default=2.0, )
    # parser.add_argument('--model_type', type=str, default='mlp', )
    parser.add_argument('--model_type', type=str, default='mlp', )
    parser.add_argument('--distance', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=1500, )
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--trainquiet', type=int, default=0, )
    args = parser.parse_args().__dict__

    # args['data_trai_n'] = 2000
    args['data_test_n'] = args['data_trai_n']
    args['vtrace'] = [args['vs'],args['ve']]
    args['batch_size'] = min(
        args['batch_size'], args['data_trai_n'], args['data_test_n'], )

    if len(args['loadfromjson']) > 1:
        import json
        loadPath = args['loadfromjson']
        strs = open(loadPath, 'r').read()
        args = json.loads(strs)

    return args

def GetParamMnistL():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='emnistL_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_mm', )
    parser.add_argument('--data_name', type=str, default='mnist', )
    parser.add_argument('--data_trai_n', type=int, default=60000, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=20, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=0.001, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list,
                        default=[-1, 600, 500, 400, 300, 200, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=1500, )
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--trainquiet', type=int, default=0, )
    args = parser.parse_args().__dict__

    args['data_test_n'] = args['data_trai_n']
    args['vtrace'] = [args['vs'],args['ve']]
    args['batch_size'] = min(
        args['batch_size'], args['data_trai_n'], args['data_test_n'], )

    if len(args['loadfromjson']) > 1:
        import json
        loadPath = args['loadfromjson']
        strs = open(loadPath, 'r').read()
        args = json.loads(strs)

    return args

def GetParamswishroll():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='swishroll_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='swishroll', )
    parser.add_argument('--data_trai_n', type=int, default=800, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=40, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, default=[-1, 600, 500, 400, 300, 200, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=1500, )
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=500 )
    parser.add_argument('--trainquiet', type=int, default=0, )
    args = parser.parse_args().__dict__

    # args['data_trai_n'] = 2000
    args['data_test_n'] = args['data_trai_n']
    args['vtrace'] = [args['vs'],args['ve']]
    args['batch_size'] = min(
        args['batch_size'], args['data_trai_n'], args['data_test_n'], )

    if len(args['loadfromjson']) > 1:
        import json
        loadPath = args['loadfromjson']
        strs = open(loadPath, 'r').read()
        args = json.loads(strs)

    return args