import importlib
experiment_1 = importlib.import_module("experiment 1")
importlib.reload(experiment_1)
# def predictions_pipeline(X_test_types, model, object_types_prediction, length_feature=1):
#     predictions = []
#     teste = []
#     for X_test_obj in X_test_types:
#         for oidx, pred_types in enumerate(predictions):
#             idx = object_types_prediction[oidx]
#             X_test_obj[:, idx + length_feature, :] = pred_types
#         test_conv_nn = X_test_obj.reshape(-1, 1, X_test_obj.shape[2], X_test_obj.shape[1])
#         pred_type = model.predict(test_conv_nn)
#         pred1_type = pred_type > 0.5
#         predictions.append(pred1_type)
#
#     return predictions,5
#
def predictions_pipeline(X_test_types, model, object_types_prediction,Y_train, length_feature=0):
    """
    Ovo je verzija kasnija, gde smo radili predikciju na pametan nacin, ne da stavimo samo
    da zauzima svugde gde je vece od 0.5
    :param X_test_types:
    :param model:
    :param object_types_prediction:
    :param Y_train:
    :param length_feature:
    :return:
    """
    predictions = []
    predictions_features = []
    for obj_idx,X_test_obj in enumerate(X_test_types):
        for oidx, pred_types in enumerate(predictions):
            idx = object_types_prediction[oidx]
            X_test_obj[:, idx + length_feature, :] = pred_types
        test_conv_nn = X_test_obj.reshape(-1, 1, X_test_obj.shape[2], X_test_obj.shape[1])
        pred_type = model.predict(test_conv_nn)
        print("shape predikcija_pre",pred_type.shape)


        pred_type = experiment_1.arrange_object(pred_type, Y_train[obj_idx])
        print("shape predikcija_ppsle", pred_type.shape)
        predictions.append(pred_type)
        predictions_features.append(X_test_obj)
    return predictions,predictions_features
