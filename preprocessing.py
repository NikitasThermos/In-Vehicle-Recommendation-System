import sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def get_preprocessor():
    def add_columns(X):
       X['distance_to_coupon'] = X['toCoupon_GEQ5min'] + X['toCoupon_GEQ15min'] + X['toCoupon_GEQ25min']
       X = X.drop(columns=['toCoupon_GEQ5min', 'toCoupon_GEQ15min', 'toCoupon_GEQ25min'])
       return X


    category_order = {'temperature': [30, 55, 80],
                    'time': ['7AM', '10AM', '2PM', '6PM', '10PM'],
                    'age': ['below21', '21', '26', '31', '36', '41', '46', '50plus'],
                    'education': ['Some High School', 'High School Graduate', 'Some college - no degree', 'Associates degree', 'Bachelors degree', 'Graduate degree (Masters or Doctorate)'],
                    'income': ['Less than $12500', '$12500 - $24999', '$25000 - $37499','$37500 - $49999', '$50000 - $62499', '$62500 - $74999', '$75000 - $87499', '$87500 - $99999', '$100000 or More'],
                    'Bar': ['never', 'less1', '1~3', '4~8', 'gt8'],
                    'CoffeeHouse': ['never', 'less1', '1~3', '4~8', 'gt8'],
                    'CarryAway': ['never', 'less1', '1~3', '4~8', 'gt8'],
                    'RestaurantLessThan20': ['never', 'less1', '1~3', '4~8', 'gt8'],
                    'Restaurant20To50': ['never', 'less1', '1~3', '4~8', 'gt8']}

    order_category_names = ['temperature', 'time', 'age',  'education', 'income', 'Bar', 'CoffeeHouse', 'CarryAway', 'RestaurantLessThan20',  'Restaurant20To50']
    non_order_category_names = ['destination', 'passanger', 'weather', 'coupon', 'expiration', 'gender', 'maritalStatus', 'occupation']
    drop = ['direction_opp', 'car']

    order_encoder = OrdinalEncoder(categories=[category_order[feature] for feature in order_category_names],
                                handle_unknown='use_encoded_value',
                                unknown_value=-1)
    add_columns_transformer = FunctionTransformer(add_columns, validate=False)


    preprocessor = ColumnTransformer(
        transformers=[
            ('ordinal', order_encoder, order_category_names),
            ('encode', OrdinalEncoder(), non_order_category_names),
            ('add_columns', add_columns_transformer, ['toCoupon_GEQ5min', 'toCoupon_GEQ15min', 'toCoupon_GEQ25min']),
            ('drop', 'drop', drop)],
        remainder='passthrough',
        verbose_feature_names_out=False).set_output(transform='pandas')
    return preprocessor