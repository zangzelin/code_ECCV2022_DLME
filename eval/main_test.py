import joblib
try:
    import eval.eval_core as eval_core
except:
    import eval_core
import numpy as np
import os
import pandas

def FindContainList(strList, conList):
    out = []
    for strs in strList:
        if conList in strs:
            out.append(strs)

    out.sort()
    return out

def TestwithLink_meta(pathinput, pathlatent):
    input_data, label_1 = joblib.load(pathinput)
    latent_data, label_2 = joblib.load(pathlatent)
    
    isinstance(label_2, np.int32)

    assert np.min(np.array(label_1) - np.array(label_2)) < 1e-5

    if 'toy_diff_std' in pathinput or "duplicate" in pathinput:
        k = label_1.shape[0]*1//2
        input_data=input_data.astype(np.float16)
        print('duplicate')
    else:
        k = label_1.shape[0]*1//20

    print('--------')
    print(pathlatent)
    print(k)
    print('--------')
    e = eval_core.Eval(
        input=input_data,
        latent=latent_data,
        label=label_2,
        k=k
        )
    
    result_dict = {
        # 'mrre-0':e.E_mrre()[0],
        # 'mrre-1':e.E_mrre()[1],
        'mrre-mean':np.mean(e.E_mrre()),
        'continuity ^':e.E_continuity(),
        'trustworthiness ^':e.E_trustworthiness(),
        'Pearson ': e.E_Rscore(),
        # 'distanceAUC ^':e.E_distanceAUC(),
        'ACCSVC':e.E_Classifacation_SVC(),
        # 'ACCKNN':e.E_Classifacation_KNN(),
        'Dismatcher':e.E_Dismatcher()
    }
    
    print(result_dict)
    return result_dict

def TestwithLink(path_s, r_all):
    pathinput = path_s+'data_label.gz'
    dirlist = os.listdir(path_s)
    listpathlatent = FindContainList(dirlist, ".png.gz")

    for pathlatent in listpathlatent:
        result_dict = TestwithLink_meta(pathinput, path_s+pathlatent)
        r = pandas.DataFrame(result_dict, index=[path_s+pathlatent], )
        
        if r_all is None:
            r_all = r
        else:
            r_all = pandas.concat([r_all, r])
        
    return r_all

if __name__ == "__main__":

    path_s = [
    'log/20210202123201_3ee4bcifa10sub_Cifa10sub_T',
    'log/20210202100845_a3d44baseline_umap_cifa10sub_Cifa10sub_T',
    'log/20210202101054_a3d44baseline_tsne_cifa10sub_Cifa10sub_T',
    ]
    r_all = None
    for p in path_s:
        r_all = TestwithLink(p+'/', r_all)
        r_all.to_csv('eval/results/{}dataset{}test.csv'.format(args['data_name'],'cifa10sub'))

    # print(r_all)
