import argparse



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
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
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

def GetParamGoogleWord():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='googleword_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='googleword', )
    parser.add_argument('--data_trai_n', type=int, default=1000000000, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=40, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, default=[-1, 100, 80, 60, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=2000, )
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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

def GetParamyj01():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='yj01_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='yj01', )
    parser.add_argument('--data_trai_n', type=int, default=8000, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=60, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, default=[-1, 500, 200, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=1000, )
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=200 )
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

def GetParamToyDuplicate():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='Duplicate_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='toy_duplicate', )
    parser.add_argument('--data_trai_n', type=int, default=900, )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=100, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100, )
    parser.add_argument('--Dec', type=bool, default=False, )
    # parser.add_argument('--NetworkStructure', type=list, default=[-1, 100, 80, 60, 2], )
    parser.add_argument('--NetworkStructure', type=list, default=[-1, 600, 500, 400, 300, 200, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )

    # train param
    parser.add_argument('--batch_size', type=int, default=1500, )
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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

def GetParamToyDiffStd():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='toy_diff_std_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='toy_diff_std', )
    parser.add_argument('--data_trai_n', type=int, default=900, )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=5, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, default=[-1, 600, 500, 400, 300, 200, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )

    # train param
    parser.add_argument('--batch_size', type=int, default=1500, )
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=1000 )
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

def GetParamScurve():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='scurve_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='scurve', )
    parser.add_argument('--data_trai_n', type=int, default=800, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=60, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    # parser.add_argument('--NetworkStructure', type=list, default=[-1, 100, 80, 60, 2], )
    parser.add_argument('--NetworkStructure', type=list, default=[-1, 100, 80, 60, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=800, )
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
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

def GetParamDigits():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='digits_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='digits', )
    parser.add_argument('--data_trai_n', type=int, default=1797, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=15, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, default=[-1, 100, 80, 60, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=1797, )
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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

def GetParamSphere5500():

    a = 10
    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='sphere5500_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='spheres5500', )
    parser.add_argument('--data_trai_n', type=int, default=5500, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=10, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, 
                        default=[-1, 100, 80, 60, 2], )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=5500, )
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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

def GetParamSphere10000():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='sphere10000_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='spheres10000', )
    parser.add_argument('--data_trai_n', type=int, default=1000, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=10, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, 
                        default=[-1, 1000, 500, 2], )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=1000, )
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
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


def GetParamgroup():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='group_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='group', )
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
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
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

def GetParamcoil20Fortime():

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
    parser.add_argument('--NetworkStructure', type=list,
                        default=[-1, 500, 400, 300, 200, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )
    # parser.add_argument('--model_type', type=str, default='mlp', )
    parser.add_argument('--model_type', type=str, default='mlp', )
    parser.add_argument('--distance', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=1440, )
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
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

def GetParamCoil100():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='coil100_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='coil100', )
    parser.add_argument('--data_trai_n', type=int, default=7200, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=10, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, default=[-1, 600, 500, 400, 300, 200, 2], )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=1500, )
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=400 )
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


def GetParamCoil100ForTime():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='coil100_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='coil100', )
    parser.add_argument('--data_trai_n', type=int, default=7200, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=10, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, 
                        default=[-1, 500, 400, 300, 200, 2], )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=720*2, )
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
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

def GetParamSMILE():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='smile_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='smile', )
    parser.add_argument('--data_trai_n', type=int, default=7200, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=80, )
    parser.add_argument('--vs', type=float, default=0.01, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, 
                        default=[-1, 100, 80, 60, 2], )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=1000, )
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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

def GetParamSMILEpluse():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='smilepluse_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='smilepluse', )
    parser.add_argument('--data_trai_n', type=int, default=7200, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=20, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, 
                        default=[-1, 100, 80, 60, 2], )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=3000, )
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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

def GetParamSumsink():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='sumsink_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='sumsink', )
    parser.add_argument('--data_trai_n', type=int, default=10000000, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=10, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, 
                        default=[-1, 500, 400, 300, 200,  2], )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=1000, )
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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


def GetParamPBMC():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='pbmc_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='pbmc', )
    parser.add_argument('--data_trai_n', type=int, default=10000000, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=float, default=300000, )
    parser.add_argument('--vs', type=float, default=0.00001, )
    parser.add_argument('--ve', type=float, default=0.00001, )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, 
                        default=[-1, 500, 400, 300, 200,  2], )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=3000, )
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
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

def GetParamMnistLSUB():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='mnistsub0_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_mm', )
    parser.add_argument('--data_name', type=str, default='mnistsub0', )
    parser.add_argument('--data_trai_n', type=int, default=60000, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=60, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=0.001, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list,
                        default=[-1, 700, 650, 600, 500, 400, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=3000, )
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100)
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

# def GetParamCifar():

#     parser = argparse.ArgumentParser(description='*** author')
#     parser.add_argument('--name', type=str, default='cifa_T', )
#     parser.add_argument('--loadfromjson', type=str, default='', )

#     # data set param
#     parser.add_argument('--method', type=str, default='LISv2_mm', )
#     parser.add_argument('--data_name', type=str, default='cifa', )
#     parser.add_argument('--data_trai_n', type=int, default=50000, )

#     # model param
#     parser.add_argument('--metric', type=str, default="euclidean", )
#     parser.add_argument('--vinput', type=float, default=100, )
#     parser.add_argument('--same_sigma', type=bool, default=True, )
#     parser.add_argument('--perplexity', type=int, default=15, )
#     parser.add_argument('--vs', type=float, default=0.001, )
#     parser.add_argument('--ve', type=float, default=100.0, )
#     parser.add_argument('--Dec', type=bool, default=False, )
#     parser.add_argument('--NetworkStructure', type=list,
#                         default=[-1, 1000, 500, 300, 2], )
#     parser.add_argument('--pow', type=float, default=2.0, )
#     parser.add_argument('--model_type', type=str, default='mlp', )

#     # train param
#     parser.add_argument('--batch_size', type=int, default=1000, )
#     parser.add_argument('--epochs', type=int, default=2000)
#     parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
#     parser.add_argument('--seed', type=int, default=1, metavar='S', )
#     parser.add_argument('--log_interval', type=int, default=100 )
#     parser.add_argument('--trainquiet', type=int, default=0, )
#     args = parser.parse_args().__dict__

#     args['data_test_n'] = args['data_trai_n']
#     args['vtrace'] = [args['vs'],args['ve']]
#     args['batch_size'] = min(
#         args['batch_size'], args['data_trai_n'], args['data_test_n'], )

#     if len(args['loadfromjson']) > 1:
#         import json
#         loadPath = args['loadfromjson']
#         strs = open(loadPath, 'r').read()
#         args = json.loads(strs)

#     return args



def GetParamFMnistL():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='FmnistL_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_mm', )
    parser.add_argument('--data_name', type=str, default='fmnist', )
    parser.add_argument('--data_trai_n', type=int, default=60000, )
    # parser.add_argument('--model_type', type=str, default='mlp', )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=10, )
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
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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

def GetParamImagenet10():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='imagenet10_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_mm', )
    parser.add_argument('--data_name', type=str, default='imagenet10', )
    parser.add_argument('--data_trai_n', type=int, default=130000, )
    # parser.add_argument('--model_type', type=str, default='mlp', )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=10, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=0.001, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list,
                        default=[-1, 600, 500, 400, 300, 200, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--model_type', type=str, default='cnn', )

    # train param
    parser.add_argument('--batch_size', type=int, default=1500, )
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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

def GetParamCifar10():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='Cifa10_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_mm', )
    parser.add_argument('--data_name', type=str, default='cifa10', )
    parser.add_argument('--data_trai_n', type=int, default=50000, )

    # model param
    parser.add_argument('--metric', type=str, default="cosine", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=50, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=0.001, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list,
                        default=[-1, 1000, 900, 800, 700, 500, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--model_type', type=str, default='cnn', )

    # train param
    parser.add_argument('--batch_size', type=int, default=1500, )
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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


def GetParamCifar10Sub():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='Cifa10sub_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_mm', )
    parser.add_argument('--data_name', type=str, default='cifa10sub', )
    parser.add_argument('--data_trai_n', type=int, default=50000, )

    # model param
    # parser.add_argument('--metric', type=str, default="chebyshev", )
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=5, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=0.001, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list,
                        default=[-1, 1000, 900, 800, 700, 500, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=1500, )
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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

def GetParamFlow():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='flow_cytometry_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_mm', )
    parser.add_argument('--data_name', type=str, default='flow_cytometry', )
    parser.add_argument('--data_trai_n', type=int, default=150000, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=10, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list,
                        default=[-1, 1000, 500, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )

    # train param
    parser.add_argument('--batch_size', type=int, default=4000, )
    parser.add_argument('--epochs', type=int, default=8001)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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

def GetParamHCL():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='HCL_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_mm', )
    parser.add_argument('--data_name', type=str, default='HCL', )
    parser.add_argument('--data_trai_n', type=int, default=600000, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=10, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list,
                        default=[-1, 5000, 2000, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )

    # train param
    parser.add_argument('--batch_size', type=int, default=4000, )
    parser.add_argument('--epochs', type=int, default=8001)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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


def GetParamSamusik():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='Samusik01_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_mm', )
    parser.add_argument('--data_name', type=str, default='Samusik01', )
    parser.add_argument('--data_trai_n', type=int, default=86864, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=20, )
    parser.add_argument('--perplexity', type=int, default=20, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list,
                        default=[-1, 100, 80, 60, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )

    # train param
    parser.add_argument('--batch_size', type=int, default=2000, )
    parser.add_argument('--epochs', type=int, default=4000)
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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


# def GetParamMnist():

#     parser = argparse.ArgumentParser(description='*** author')
#     parser.add_argument('--name', type=str, default='mnist_T', )
#     parser.add_argument('--loadfromjson', type=str, default='', )

#     # data set param
#     parser.add_argument('--method', type=str, default='LISv2_mm', )
#     parser.add_argument('--data_name', type=str, default='mnist', )
#     parser.add_argument('--data_trai_n', type=int, default=8000, )

#     # model param
#     parser.add_argument('--metric', type=str, default="euclidean", )
#     parser.add_argument('--vinput', type=float, default=100, )
#     parser.add_argument('--same_sigma', type=bool, default=12, )
#     parser.add_argument('--perplexity', type=int, default=12, )
#     parser.add_argument('--vs', type=float, default=0.001, )
#     parser.add_argument('--ve', type=float, default=100.0, )
#     parser.add_argument('--Dec', type=bool, default=False, )
#     parser.add_argument('--NetworkStructure', type=list,
#                         default=[-1, 1000, 500, 2], )
#     parser.add_argument('--pow', type=float, default=2.0, )

#     # train param
#     parser.add_argument('--batch_size', type=int, default=800, )
#     parser.add_argument('--epochs', type=int, default=800)
#     parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
#     parser.add_argument('--seed', type=int, default=1, metavar='S', )
#     parser.add_argument('--log_interval', type=int, default=100 )
#     parser.add_argument('--trainquiet', type=int, default=0, )
#     args = parser.parse_args().__dict__

#     # args['data_trai_n'] = 2000
#     args['data_test_n'] = args['data_trai_n']
#     args['vtrace'] = [args['vs'],args['ve']]
#     args['batch_size'] = min(
#         args['batch_size'], args['data_trai_n'], args['data_test_n'], )

#     if len(args['loadfromjson']) > 1:
#         import json
#         loadPath = args['loadfromjson']
#         strs = open(loadPath, 'r').read()
#         args = json.loads(strs)

#     return args

def GetParamCora():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='cora_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_mm', )
    parser.add_argument('--data_name', type=str, default='cora', )
    parser.add_argument('--data_trai_n', type=int, default=2708, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=2, )
    parser.add_argument('--perplexity', type=int, default=2, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list,
                        default=[-1, 100, 80, 60, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )

    # train param
    parser.add_argument('--batch_size', type=int, default=2708, )
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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


def GetParamseveredsphere():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='severedsphere_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='severedsphere', )
    parser.add_argument('--data_trai_n', type=int, default=879, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=20, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, default=[16384 * 3, 500, 500, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )
    parser.add_argument('--model_type', type=str, default='mlp', )

    # train param
    parser.add_argument('--batch_size', type=int, default=100, )
    parser.add_argument('--epochs', type=int, default=800)
    # parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=1, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
    parser.add_argument('--trainquiet', type=int, default=0, )
    args = parser.parse_args().__dict__

    # args['data_trai_n'] = 2000
    args['data_test_n'] = args['data_trai_n']
    args['vtrace'] = [args['vs'],args['ve']]
    args['batch_size'] = min(
        args['batch_size'], args['data_trai_n'], args['data_test_n'], )
    args['batch_size'] = 912

    if len(args['loadfromjson']) > 1:
        import json
        loadPath = args['loadfromjson']
        strs = open(loadPath, 'r').read()
        args = json.loads(strs)

    return args

def GetParamPbmc3k():

    parser = argparse.ArgumentParser(description='*** author')
    parser.add_argument('--name', type=str, default='pbmc3k_T', )
    parser.add_argument('--loadfromjson', type=str, default='', )

    # data set param
    parser.add_argument('--method', type=str, default='LISv2_m', )
    parser.add_argument('--data_name', type=str, default='pbmc3k', )
    parser.add_argument('--data_trai_n', type=int, default=2638, )

    # model param
    parser.add_argument('--metric', type=str, default="euclidean", )
    parser.add_argument('--vinput', type=float, default=100, )
    parser.add_argument('--same_sigma', type=bool, default=True, )
    parser.add_argument('--perplexity', type=int, default=10, )
    parser.add_argument('--vs', type=float, default=0.001, )
    parser.add_argument('--ve', type=float, default=100.0, )
    parser.add_argument('--Dec', type=bool, default=False, )
    parser.add_argument('--NetworkStructure', type=list, default=[-1, 100, 80, 60, 2], )
    parser.add_argument('--pow', type=float, default=2.0, )

    # train param
    parser.add_argument('--batch_size', type=int, default=2638, )
    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', )
    parser.add_argument('--seed', type=int, default=2, metavar='S', )
    parser.add_argument('--log_interval', type=int, default=100 )
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











