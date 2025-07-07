import os
import unittest

import pandas as pd
import requests
import numpy as np
import json
from service.v4.protocol import features_names
from path_utils import cache_root_de
from service.v5.protocol import get_config

day_milliseconds = 24 * 60 * 60 * 1000

configs = get_config()


class TestPosts(unittest.TestCase):
    port = 3710

    def setUp(self):
        port = TestPosts.port
        host = 'http://localhost'
        self.send_post = lambda data: requests.post('{}:{}/predict'.format(host, port), data=json.dumps(data),
                                                    headers={'Content-Type': 'application/json'})
        self.send_post_outcome = lambda data: requests.post('{}:{}/predict/outcome'.format(host, port),
                                                       data=json.dumps(data),
                                                       headers={'Content-Type': 'application/json'})
        self.send_post_dl = lambda data: requests.post('{}:{}/predict/outcome/dl'.format(host, port),
                                                       data=json.dumps(data),
                                                       headers={'Content-Type': 'application/json'})
        self.send_post_xgb = lambda data: requests.post('{}:{}/predict/outcome/xgb'.format(host, port),
                                                        data=json.dumps(data),
                                                        headers={'Content-Type': 'application/json'})
        self.send_post_event = lambda data: requests.post('{}:{}/predict/event/xgb'.format(host, port),
                                                          data=json.dumps(data),
                                                          headers={'Content-Type': 'application/json'})
        self.send_post_variable = lambda data: requests.post('{}:{}/tools/parse_variables'.format(host, port),
                                                             data=json.dumps(data),
                                                             headers={'Content-Type': 'application/json'})
        self.send_get = lambda data: requests.get('{}:{}/'.format(host, port), data=json.dumps(data),
                                                  headers={'Content-Type': 'application/json'})

    def test_connection(self):
        data = {'test': 'GET'}
        result = self.send_get(data)
        self.assertEqual(result.ok, True)
        self.assertEqual(result.text, 'GET request for /')

    def test_post_case1(self):
        from test_cases import test_case1 as test_case
        # self.assertEqual(np.all([fn in test_case for fn in features_names]), True)

        data = test_case.copy()
        data['idx'] = 0

        for k in data:
            if 'time_trop' in k:
                data[k] *= 1000. * 60. * 60. * 24.

        data.pop('time_trop7')
        data.pop('trop7')

        result = self.send_post_outcome(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' in response_dict, True)

        result = self.send_post_variable(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)

        result = self.send_post_dl(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual('p5_pred_dl' in response_dict, True)

        result = self.send_post_xgb(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual('p3_pred_xgb' in response_dict, True)

        result = self.send_post_event(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual('event_dmi30d_pred_xgb' in response_dict, True)
        # response_dict = json.loads(result.text.replace("'", '"'))
        # for k in known_response:
        #     self.assertEqual(known_response[k], response_dict[k])

    def test_send_unused_features(self):
        from test_cases import test_case1 as test_case

        data = test_case.copy()
        data['idx'] = 0

        for k in data:
            if 'time_trop' in k:
                data[k] *= 1000. * 60. * 60. * 24.

        result = self.send_post_variable(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))

        unused_keys = ['Unnamed: 0', 'cohort_id', 'ds', 'supercell_id', 'subjectid',
         'avgtrop', 'avgspd', 'maxtrop', 'mintrop', 'maxvel', 'minvel', 'divtrop', 'difftrop', 'diffvel', 'logtrop0',
         'out5', 'out3c', 'outl1', 'outl2', 'event_mi', 'event_t1mi', 'event_t2mi', 'event_t4mi', 'event_t5mi',
         'event_dead', 'event_dmi30d',
         'quantized_trop_0-2', 'quantized_trop_2-4', 'quantized_trop_4-6', 'quantized_trop_6-8', 'quantized_trop_8-10',
         'quantized_trop_10-12', 'quantized_trop_12-14', 'quantized_trop_14-16', 'quantized_trop_16-18',
         'quantized_trop_18-20', 'quantized_trop_20-22', 'quantized_trop_22-24', 'set', 'idx']

        # unused_keys += configs['features']['exclude']['data3']

        self.assertEqual(set(response_dict['unused_query_keys']) == set(unused_keys), True)

    def test_randomly_omitted_post_case1(self):
        from test_cases import test_case1 as test_case
        # self.assertEqual(np.all([fn in test_case for fn in features_names]), True)

        # with 50% features
        ratio = 0.5
        random_keys = [features_names[i] for i in np.random.permutation(len(features_names))][:int(len(features_names)*ratio)]

        data = {k: test_case[k] for k in test_case if k in random_keys}
        data['idx'] = 0
        for k in data:
            if 'time_trop' in k:
                data[k] *= 1000. * 60. * 60. * 24.
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual(np.all([k not in data for k in response_dict['unmatched_query_keys']]), True)
        self.assertEqual(np.all([k in data for k in response_dict['matched_query_dict'].keys()]), True)

        # with one random feature
        random_keys = [features_names[i] for i in np.random.permutation(len(features_names))][0]

        data = {k: test_case[k] for k in test_case if k in random_keys}
        data['idx'] = 0
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual(np.all([k not in data for k in response_dict['unmatched_query_keys']]), True)
        self.assertEqual(np.all([k in data for k in response_dict['matched_query_dict'].keys()]), True)

    def test_no_idx_post_case1(self):
        from test_cases import test_case1 as test_case
        # self.assertEqual(np.all([fn in test_case for fn in features_names]), True)

        data = test_case.copy()
        for k in data:
            if 'time_trop' in k:
                data[k] *= 1000. * 60. * 60. * 24.
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # for k in known_response:
        #     self.assertEqual(known_response[k], response_dict[k])
        self.assertEqual('error_message' in response_dict, True)

    def test_null_post(self):
        test_case = {fn: 'nan' for fn in features_names}

        data = test_case.copy()
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        # for k in known_response:
        #     self.assertEqual(known_response[k], response_dict[k])
        self.assertEqual('warning_message' in response_dict, True)

    def test_age_only_varied_post(self):

        data = dict()
        data['gender'] = 1
        for age in range(10, 90, 10):
            data['age'] = age
            print(data)
            result = self.send_post(data)
            self.assertEqual(result.ok, True)
            response_dict = json.loads(result.text.replace("'", '"'))
            print(response_dict)
            # for k in known_response:
            #     self.assertEqual(known_response[k], response_dict[k])
            if age < 18 or age > 115:
                self.assertEqual('error_message' in response_dict, True)
            else:
                self.assertEqual('error_message' not in response_dict, True)

    def test_single_trop_post(self):

        data = {
                "age": 50,
                "gender": 0,
                "angiogram": 0,
                "trop1": 20,
                "time_trop1": 0.20 * day_milliseconds,
        } 

        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict)
        self.assertEqual('error_message' not in response_dict, True)

    def test_nil_trop_post(self):

        data = {
                "age": 50,
                "gender": 0,
                "angiogram": 0,
        }

        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict)
        self.assertEqual('error_message' not in response_dict, True)

    def test_two_digits_trop_post(self):

        data = {
                "age": 50,
                "gender": 0,
                "angiogram": 0,
                "trop1": 20,
                "time_trop1": 0.20 * day_milliseconds,
                "trop11": 20,
                "time_trop11": 0.20 * day_milliseconds,
        }

        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict)
        self.assertEqual('error_message' not in response_dict, True)

    def test_trop_time_large_out_of_bound_post(self):

        data = {
                "age": 50,
                "gender": 0,
                "angiogram": 0,
                "trop11": 20,
                "time_trop11": 2 * day_milliseconds,
        }

        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict)
        self.assertEqual('error_message' in response_dict, True)

        data = {
            "age": 50,
            "gender": 0,
            "angiogram": 0,
            "trop1": 20,
            "time_trop1": 0.5 * day_milliseconds,
            "trop11": 20,
            "time_trop11": 2 * day_milliseconds,
        }

        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict)
        self.assertEqual('warning_message' in response_dict, True)

        data = {
                "age": 50,
                "gender": 0,
                "angiogram": 0,
                "trop11": 20,
                "time_trop11": -0.1 * day_milliseconds,
        }

        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict)
        self.assertEqual('error_message' in response_dict, True)

    def test_trop_time_naming_mismatch_post(self):

        data = {
                "age": 50,
                "gender": 0,
                "angiogram": 0,
                "trop1": 20,
                "time_trop11": 0.2 * day_milliseconds,
        }

        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict)
        self.assertEqual('error_message' in response_dict, True)

    def test_trop_order_post(self):

        data = {
                "age": 50,
                "gender": 0,
                "angiogram": 0,
                "trop1": 3,
                "trop2": 3,
                "trop11": 3,
                "trop21": 4,
                "time_trop1": 0.1 * day_milliseconds,
                "time_trop2": 0.2 * day_milliseconds,
                "time_trop11": 0.3 * day_milliseconds,
                "time_trop21": 0.4 * day_milliseconds,
        }

        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict)
        self.assertEqual('error_message' not in response_dict, True)

    def test_trop_lower_bound_post(self):

        data = {
                "age": 50,
                "gender": 0,
                "angiogram": 0,
                "trop1": 2.99,
                "time_trop1": 0.4 * day_milliseconds,
        }

        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict)
        self.assertEqual('warning_message' in response_dict, True)

        data = {
                "age": 50,
                "gender": 0,
                "angiogram": 0,
                "trop1": -0.01,
                "time_trop1": 0.4 * day_milliseconds,
        }

        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict)
        self.assertEqual('error_message' in response_dict, True)

    def test_lukah_t1mi_case(self):
        data = {
                "age": 70,
                "gender": 1,
                "angiogram": 1,
                "trop1": 30,
                "time_trop1": 3600000,
                "trop2": 200,
                "time_trop2": 7200000,
        }

        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict)
        self.assertEqual(response_dict['l1_pred_xgb'], 1)
        self.assertEqual(response_dict['l2_pred_xgb'], 1)
        self.assertEqual(response_dict['p3_pred_xgb'], 2)

        data = {
                "age": 60,
                "gender": 0,
                "angiogram": 0,
                "trop1": 5,
                "time_trop1": 3600000,
                "trop2": 10,
                "time_trop2": 7200000,
                "outcome_model_thld_method_xgb": "tpr"
        }

        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict)
        self.assertEqual(response_dict['l1_pred_xgb'], 0)
        self.assertEqual(response_dict['l2_pred_xgb'], 0)
        self.assertEqual(response_dict['p3_pred_xgb'], 0)

    def test_joey_wbc_case(self):
        data = {"idPatient": [40876], "idEvent": [4527856], "age": [67.95131278538813], "gender": [0.0], "angiogram": [0.0],
         "time_trop1": [0.0], "trop1": [69.0], "time_trop2": [31500000.0], "trop2": [329.0], "phys_albumin": [19.0],
         "phys_bnp": [1753.0], "phys_creat": [112.0], "phys_crp": [286.0], "phys_haeglob": [147.0], "phys_lacta": [1.8],
         "phys_lactv": [2.1], "phys_pco2": [30.0], "phys_ph": [7.38], "phys_platec": [17.0], "phys_platev": [9.3],
         "phys_po2": [87.0], "phys_tsh": [0.72], "phys_urea": [8.4], "phys_urate": [0.28], "phys_wbc": [0.0],
         "mdrd_gfr": [44.0], "prioracs": [0.0], "priorami": [0.0], "priorcabg": [0.0], "priorcopd": [1.0],
         "priorcva": [0.0], "priordiab": [0.0], "priorhf": [0.0], "priorhtn": [0.0], "priorhyperlipid": [0.0],
         "priorpci": [0.0], "priorrenal": [0.0], "priorsmoke": [1.0]}

        for k in data:
            data[k] = data[k][0]
        data['phys_wbc'] = 0.01
        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict)
        self.assertNotEqual(str(response_dict['l1_prob_dl']), 'nan')
        self.assertNotEqual(str(response_dict['l2_prob_dl']), 'nan')

    # data file comes from Joey on 13/07/2022, in the email thread 'Question'
    def test_joy_13072022(self):
        data = pd.read_csv(os.path.join(cache_root_de, 'diff_joy_13072022.csv'))
        data = dict(data.loc[0])

        # filling the gender value as otherwise it fails the rule "age and gender" should be present.
        data['gender'] = 0

        data['time_trop2'] = data['time_trop2'] - 100000
        data['time_trop1'] = data['time_trop1'] - 100000
        data = {k: str(v) for k, v in data.items()}
        print(data)
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        print(response_dict['event_dmi30d_prob_xgb'])
        self.assertNotEqual(str(response_dict['l1_prob_dl']), 'nan')
        self.assertNotEqual(str(response_dict['l2_prob_dl']), 'nan')

    def test_control(self):
        # all control variables given
        data = {'age': 30, 'gender': 0, 'recomm_outcome_model_type_dl': 'dl', 'recomm_outcome_version_dl': 'v5',
                'recomm_outcome_model_type_xgb': 'xgb', 'recomm_outcome_version_xgb': 'v5',
                'recomm_event_model_type': 'xgb', 'recomm_event_version': 'v5',
                'event_model_thld_method_xgb': 'roc',
                'outcome_model_thld_method_xgb': 'roc',
                'outcome_model_thld_method_dl': 'roc',
                'model_idx': -1}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual('control_message' not in response_dict, True)

        # no control variables given
        data = {'age': 30, 'gender': 0}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual('control_message' in response_dict, True)

        # control variable type cast fails
        data = {'age': 30, 'gender': 0, 'recomm_outcome_model_type_dl': 'dl', 'recomm_outcome_version_dl': 'v5',
                'recomm_event_model_type': 'xgb', 'recomm_event_version': 'v5', 'model_idx': 'not a number'}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' in response_dict, True)
        self.assertEqual('control_message' not in response_dict, True)

        # control variables out of the predefined options
        data = {'age': 30, 'gender': 0, 'recomm_outcome_model_type_dl': 'foo', 'recomm_outcome_version_dl': 'v5',
                'recomm_event_model_type': 'xgb', 'recomm_event_version': 'v5', 'model_idx': -1}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' in response_dict, True)
        self.assertEqual('control_message' not in response_dict, True)

    def test_recomm(self):
        # check v3 dl model usage
        data = {'age': 30, 'gender': 0,
                'recomm_outcome_model_type_dl': 'dl', 'recomm_outcome_version_dl': 'v3',
                'recomm_outcome_model_type_xgb': 'xgb', 'recomm_outcome_version_xgb': 'v5',
                'recomm_event_model_type': 'xgb', 'recomm_event_version': 'v5',
                'event_model_thld_method_xgb': 'roc',
                'outcome_model_thld_method_xgb': 'roc',
                'outcome_model_thld_method_dl': 'roc',
                'model_idx': -1}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' in response_dict, False)
        self.assertEqual('control_message' in response_dict, False)
        self.assertEqual('recomm_text' in response_dict, True)

        # check v3 xgb model usage
        data = {'age': 30, 'gender': 0,
                'recomm_outcome_model_type_dl': 'dl', 'recomm_outcome_version_dl': 'v5',
                'recomm_outcome_model_type_xgb': 'xgb', 'recomm_outcome_version_xgb': 'v3',
                'recomm_event_model_type': 'xgb', 'recomm_event_version': 'v5',
                'event_model_thld_method_xgb': 'roc',
                'outcome_model_thld_method_xgb': 'roc',
                'outcome_model_thld_method_dl': 'roc',
                'model_idx': -1,
                'lhn': 'debug_lhn', 'heart_score': 3}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' in response_dict, False)
        self.assertEqual('control_message' in response_dict, False)
        self.assertEqual('recomm_text' in response_dict, True)
        print({k: response_dict[k] for k in response_dict.keys() if 'recomm' in k})

    def test_bounds(self):

        # check age bound
        data = {'age': 17.99, 'gender': 0}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' in response_dict, True)

        data = {'age': 115.01, 'gender': 0}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('warning_message' in response_dict, True)

        # check age in bound
        data = {'age': 18, 'gender': 0}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual('warning_message' in response_dict, True)

        # check heart score = 0 but inclusive = False
        data = {'age': 18, 'gender': 1, 'heart_score': 0}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' in response_dict, True)

        # missing age
        data = {'age': 'nan', 'gender': 1}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('warning_message' in response_dict, True)

        # missing gender
        data = {'age': '38', 'gender': 'nan'}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('warning_message' in response_dict, True)

        # missing age and gender
        data = {}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('warning_message' in response_dict, True)

        # type cast
        data = {'age': 'eighteen', 'gender': '1'}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' in response_dict, True)

        data = {'age': '40', 'gender': 'male'}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' in response_dict, True)

        # out of options
        data = {'age': '40', 'gender': '2'}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' in response_dict, True)

        # lhn cast
        data = {'age': 114, 'gender': '1', 'lhn': '10'}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual('warning_message' in response_dict, True)

        # warning lower bound treat as missing
        data = {'age': 114, 'gender': '1', 'phys_bnp': 49}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual('warning_message' in response_dict, True)

        # warning lower bound treat as missing
        data = {'age': 114, 'gender': '1', 'phys_creat': -0.01}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual('warning_message' in response_dict, True)

    def test_recommend_function(self):
        from service.v5.recomm_shell import RecommShellModel
        from path_utils import model_root
        r = RecommShellModel(
            recomm_path=os.path.join(model_root, 'v5', 'rapidx_recommendations_v4.csv'),
            conv_path=os.path.join(model_root, 'v5', 'outcome_conversion.csv'),
            lhn_path=os.path.join(model_root, 'v5', 'lhn_classification.csv'))
        control_variables = {'recomm_outcome_model_type_dl': 'dl', 'recomm_outcome_version_dl': 'v5',
                             'recomm_outcome_model_type_xgb': 'xgb', 'recomm_outcome_version_xgb': 'v5',
                             'recomm_event_model_type': 'xgb', 'recomm_event_version': 'v5'}
        features = {'lhn': 'debug_lhn', 'heart_score': 3.1}
        outcome_dl = {'p5_prob_dl': [0.1, 0.2, 0.3, 0.35, 0.05], 'p5_pred_dl': 3}
        outcome_xgb = {'p3_pred_xgb': 0}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 't2mi_acute_text')

        features = {'lhn': 'debug_lhn', 'heart_score': 3.1}
        outcome_dl = {'p5_prob_dl': [0.1, 0.2, 0.35, 0.3, 0.05], 'p5_pred_dl': 2}
        outcome_xgb = {'p3_pred_xgb': 0}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 't2mi_acute_text')

        features = {'lhn': 'debug_lhn', 'heart_score': 3.1}
        outcome_dl = {'p5_prob_dl': [0.1, 0.35, 0.2, 0.3, 0.05], 'p5_pred_dl': 1}
        outcome_xgb = {'p3_pred_xgb': 0}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 'chronic_text')
        # self.assertEqual(result['recomm_prob'], 0.35)

        features = {'lhn': 'debug_lhn', 'heart_score': 3.1}
        outcome_dl = {'p5_prob_dl': [0.1, 0.2, 0.05, 0.3, 0.35], 'p5_pred_dl': 4}
        outcome_xgb = {'p3_pred_xgb': 0}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 't1mi_text')

        features = {'lhn': 'debug_lhn', 'heart_score': 3.1}
        outcome_dl = {'p5_prob_dl': [0.35, 0.2, 0.1, 0.3, 0.05], 'p5_pred_dl': 0}
        outcome_xgb = {'p3_pred_xgb': 0}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 'Normal_>0.01and>3')
        # self.assertEqual(result['recomm_prob'], 0.35)

        features = {'lhn': 'debug_lhn', 'heart_score': 3.1}
        outcome_dl = {'p5_prob_dl': [0.35, 0.2, 0.1, 0.3, 0.05], 'p5_pred_dl': 0}
        outcome_xgb = {'p3_pred_xgb': 0}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 0}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 'Normal_>0.01and>3')

        features = {'lhn': 'debug_lhn', 'heart_score': 3}
        outcome_dl = {'p5_prob_dl': [0.35, 0.2, 0.1, 0.3, 0.05], 'p5_pred_dl': 0}
        outcome_xgb = {'p3_pred_xgb': 0}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 0}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 'Normal_<=0.01and<=3')

        features = {'lhn': 'debug_lhn', 'heart_score': 2.9}
        outcome_dl = {'p5_prob_dl': [0.35, 0.2, 0.1, 0.3, 0.05], 'p5_pred_dl': 0}
        outcome_xgb = {'p3_pred_xgb': 0}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 0}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 'Normal_<=0.01and<=3')

        features = {'lhn': 'debug_lhn', 'heart_score': 2.9}
        outcome_dl = {'p5_prob_dl': [0.35, 0.2, 0.1, 0.3, 0.05], 'p5_pred_dl': 0}
        outcome_xgb = {'p3_pred_xgb': 0}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 'Normal_>0.01and>3')
        # self.assertEqual(result['recomm_prob'], 0.35)

        features = {'lhn': 'debug_lhn', 'heart_score': 2.9}
        outcome_dl = {'p5_prob_dl': [0.35, 0.2, 0.1, 0.3, 0.05], 'p5_pred_dl': 0}
        outcome_xgb = {'p3_pred_xgb': 2}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 't1mi_text')

        features = {'lhn': 'debug_lhn', 'heart_score': 2.9}
        outcome_dl = {'p5_prob_dl': [0.35, 0.2, 0.1, 0.3, 0.05], 'p5_pred_dl': 0}
        outcome_xgb = {'p3_pred_xgb': 1}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 't2mi_acute_text')

        features = {'lhn': 'debug_lhn', 'heart_score': 2.9}
        outcome_dl = {'p5_prob_dl': [0.05, 0.2, 0.1, 0.3, 0.35], 'p5_pred_dl': 4}
        outcome_xgb = {'p3_pred_xgb': 1}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 't1mi_text')

        features = {'lhn': 'debug_lhn', 'heart_score': 2.9}
        outcome_dl = {'p5_prob_dl': [0.05, 0.35, 0.1, 0.3, 0.2], 'p5_pred_dl': 1}
        outcome_xgb = {'p3_pred_xgb': 2}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 't1mi_text')
        # self.assertEqual(result['recomm_prob'], 0.2)

        features = {'lhn': 'debug_lhn', 'heart_score': 2.9}
        outcome_dl = {'p5_prob_dl': [0.2, 0.1, 0.35, 0.3, 0.05], 'p5_pred_dl': 2}
        outcome_xgb = {'p3_pred_xgb': 1}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 't2mi_acute_text')

        features = {'lhn': 'debug_lhn', 'heart_score': 2.9}
        outcome_dl = {'p5_prob_dl': [0.5, 0.1, 0.36, 0.29, 0.05], 'p5_pred_dl': 2}
        outcome_xgb = {'p3_pred_xgb': 1}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 't2mi_acute_text')
        # self.assertEqual(result['recomm_prob'], 0.36)

        features = {'lhn': 'debug_lhn', 'heart_score': 2.9}
        outcome_dl = {'p5_prob_dl': [0.5, 0.1, 0.29, 0.36, 0.05], 'p5_pred_dl': 2}
        outcome_xgb = {'p3_pred_xgb': 1}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual(result['recomm_text'], 't2mi_acute_text')
        # self.assertEqual(result['recomm_prob'], 0.36)

        features = {'lhn': 'salhn', 'heart_score': 2.9}
        outcome_dl = {'p5_prob_dl': [0.5, 0.1, 0.29, 0.36, 0.05], 'p5_pred_dl': 2}
        outcome_xgb = {'p3_pred_xgb': 1}
        event_xgb = {'event_dmi30d_above_1pc_pred_xgb': 1}
        result = r.inference_single(features=features, control_variables=control_variables,
                                    response_outcome_dl=outcome_dl, response_outcome_xgb=outcome_xgb,
                                    response_event=event_xgb)
        self.assertEqual('Consider and clinically exclude Aortic Dissection and Pulmonary Embolism' in
                         result['recomm_text'], True)
        # self.assertEqual(result['recomm_prob'], 0.36)

    def test_threshold_change_function(self):
        data = {'age': 30, 'gender': 0, 'outcome_model_thld_method_xgb': 'default', 'model_idx': 0}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual(response_dict['l1_thld_xgb'], 0.5)
        self.assertEqual(response_dict['l2_thld_xgb'], 0.5)

        data = {'age': 30, 'gender': 0, 'model_idx': 0}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertNotEqual(response_dict['l1_thld_xgb'], 0.5)
        self.assertNotEqual(response_dict['l2_thld_xgb'], 0.5)

        data = {'age': 30, 'gender': 0, 'model_idx': 0, 'outcome_model_thld_method_dl': 'default'}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual(response_dict['l1_thld_dl'], 0.5)
        self.assertEqual(response_dict['l2_thld_dl'], 0.5)

        data = {'age': 30, 'gender': 0, 'model_idx': -1, 'outcome_model_thld_method_dl': 'default'}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual(set(response_dict['l1_thld_dl']), {0.5})
        self.assertEqual(set(response_dict['l2_thld_dl']), {0.5})

        data = {'age': 30, 'gender': 0, 'model_idx': 0}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertNotEqual(response_dict['l1_thld_dl'], 0.5)
        self.assertNotEqual(response_dict['l2_thld_dl'], 0.5)

        data = {'age': 30, 'gender': 0, 'outcome_model_thld_method_xgb': 'tpr'}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual(response_dict['outcome_threshold_method_xgb'], 'tpr')

        data = {'age': 30, 'gender': 0, 'event_model_thld_method_xgb': 'pr'}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual(response_dict['event_dmi30d_threshold_method'], 'pr')

        data = {'age': 30, 'gender': 0, 'outcome_model_thld_method_dl': 'default'}
        data = {k: str(v) for k, v in data.items()}
        result = self.send_post(data)
        self.assertEqual(result.ok, True)
        response_dict = json.loads(result.text.replace("'", '"'))
        self.assertEqual('error_message' not in response_dict, True)
        self.assertEqual(response_dict['outcome_threshold_method_dl'], 'default')

    def test_development_deployment_consistency_outcome_xgb(self):
        from path_utils import cache_root_d3 as cache_root

        case_limit = 10

        # n_boots = 50
        split_method = 'cv'
        angio_or_ecg = 'ecg'
        # threshold_method = 'tpr'
        label_name = 'adjudicatorDiagnosis'
        model_folder = f'xgb_{label_name}_{split_method}_{angio_or_ecg}_v5.1.2'

        save_path = os.path.join(cache_root, f'{model_folder}/sample.npy')
        saved_data = np.load(save_path, allow_pickle=True).item()
        # df_outbag = saved_data['df_outbag']
        outbag_idxs = saved_data['outbag_idxs']

        df = pd.read_csv(os.path.join(cache_root, 'data_raw_trop6_phys_0.csv'), low_memory=False)
        # df = df_outbag
        df = df.iloc[outbag_idxs]
        df = df.reset_index(drop=True)

        found_cases = 0
        for case_no, (r_idx, row) in enumerate(df.iterrows()):
            if not row['dataset'] in ['data3', 'data_ecg']:
                continue

            if found_cases >= case_limit:
                break
            processed_data = df.iloc[case_no]
            processed_result = (saved_data['y1_prob'][case_no], saved_data['y2_prob'][case_no])
            df_data3 = pd.read_csv(os.path.join(cache_root, 'data_v3.csv'))
            raw_data = dict(df_data3[df_data3['idPatient'] == processed_data['idPatient']].iloc[0])
            # self.assertEqual(processed_data['idPatient'], raw_data['idPatient'])

            data = {k: str(v) for k, v in raw_data.items() if not pd.isna(v)}
            result = self.send_post(data)
            self.assertEqual(result.ok, True)
            response_dict = json.loads(result.text.replace("'", '"'))
            deployment_result = (response_dict['l1_prob_xgb'][0], response_dict['l2_prob_xgb'][0])
            self.assertAlmostEqual(processed_result[0], deployment_result[0], places=6)
            self.assertAlmostEqual(processed_result[1], deployment_result[1], places=6)

            found_cases += 1

    def test_development_deployment_consistency_event_xgb(self):
        from path_utils import cache_root_d3 as cache_root

        case_limit = 10

        # n_boots = 50
        split_method = 'cv'
        angio_or_ecg = 'none'
        # threshold_method = 'tpr'
        label_name = 'event_dmi30d'
        model_folder = f'xgb_sklearn_{label_name}_{split_method}_{angio_or_ecg}_v5.1.2'

        save_path = os.path.join(cache_root, f'{model_folder}/sample.npy')
        saved_data = np.load(save_path, allow_pickle=True).item()
        # df_outbag = saved_data['df_outbag']
        outbag_idxs = saved_data['outbag_idxs']

        df = pd.read_csv(os.path.join(cache_root, 'data_raw_trop6_phys_0.csv'), low_memory=False)
        # df = df_outbag
        df = df.iloc[outbag_idxs]
        df = df.reset_index(drop=True)

        found_cases = 0
        for case_no, (r_idx, row) in enumerate(df.iterrows()):
            if not row['dataset'] in ['data3', 'data_ecg']:
                continue

            if found_cases >= case_limit:
                break
            processed_data = df.iloc[case_no]
            processed_result = saved_data['y1_prob'][case_no]
            df_data3 = pd.read_csv(os.path.join(cache_root, 'data_v3.csv'))
            raw_data = dict(df_data3[df_data3['idPatient'] == processed_data['idPatient']].iloc[0])
            # self.assertEqual(processed_data['idPatient'], raw_data['idPatient'])

            data = {k: str(v) for k, v in raw_data.items() if not pd.isna(v)}
            result = self.send_post(data)
            self.assertEqual(result.ok, True)
            response_dict = json.loads(result.text.replace("'", '"'))
            deployment_result = response_dict['event_dmi30d_prob_xgb'][0]
            self.assertAlmostEqual(processed_result, deployment_result, places=6)

            found_cases += 1

    def test_development_deployment_consistency_outcome_dl(self):
        from path_utils import cache_root_d3 as cache_root

        case_limit = 10

        use_ecg = True
        model_folder = f'outcome_data3_lm1_lr5e-3_use_ecg_{use_ecg}_b128_v5.1.1'

        save_path = os.path.join(cache_root, f'{model_folder}/sample.npy')
        saved_data = np.load(save_path, allow_pickle=True).item()
        # df_outbag = saved_data['df_outbag']
        outbag_idxs = saved_data['outbag_idxs']

        df = pd.read_csv(os.path.join(cache_root, 'data_raw_trop6_phys_0.csv'), low_memory=False)
        # df = df_outbag
        df = df.iloc[outbag_idxs]
        df = df.reset_index(drop=True)

        found_cases = 0
        for case_no, (r_idx, row) in enumerate(df.iterrows()):
            if not row['dataset'] in ['data3', 'data_ecg']:
                continue

            if found_cases >= case_limit:
                break
            processed_data = df.iloc[case_no]
            processed_result = (saved_data['y1_prob'][case_no], saved_data['y2_prob'][case_no])
            df_data3 = pd.read_csv(os.path.join(cache_root, 'data_v3.csv'))
            raw_data = dict(df_data3[df_data3['idPatient'] == processed_data['idPatient']].iloc[0])
            # self.assertEqual(processed_data['idPatient'], raw_data['idPatient'])

            data = {k: str(v) for k, v in raw_data.items() if not pd.isna(v)}
            result = self.send_post(data)
            self.assertEqual(result.ok, True)
            response_dict = json.loads(result.text.replace("'", '"'))
            deployment_result = (response_dict['l1_prob_dl'][0], response_dict['l2_prob_dl'][0])
            self.assertAlmostEqual(processed_result[0], deployment_result[0], places=6)
            self.assertAlmostEqual(processed_result[1], deployment_result[1], places=6)

            found_cases += 1


if __name__ == '__main__':

    from sys import argv

    if len(argv) == 2:
        TestPosts.port = int(argv[1])

    unittest.main(argv=['first-arg-is-ignored'], exit=False)
