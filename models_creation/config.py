class Config:
    """
    This class contains all the parameters of the program.

    Attributes:
        - PATH (str): path to the database
        - AAMI_NAME_FILE (str): name of the file containing the AAMI categories
        - ALL_NAME_FILE (str): name of the file containing all the categories
        - BEFORE_R_PEAK (int): number of points before the R peak
        - AFTER_R_PEAK (int): number of points after the R peak
        - TIME_LENGTH (int): total number of points in the ECG
        - INTER_POINTS_NUMBER (int): number of points between each point of the ECG (P,Q,R,S,T)
        - TOTAL_POINTS_NUMBER (int): number of points in the ECG
        - THRESHOLD (float): threshold for the activation function
        - NEURON_NUMBER (int): number of different neurons (amplitude)
        - AAMI_categories (list): list of the AAMI categories
        - AAMI_categories_to_supp (list): list of the AAMI categories to not use
        - ALL_categories (list): list of all the categories
        - ALL_categories_to_supp (list): list of all the categories to not use
        - KERNEL_SMOOTH_SIZE (int): size of the kernel for the smoothing
        - coloscale (list): list of colors
    """
    PATH = "models_creation/data"
    AAMI_NAME_FILE = 'mitdb.pkl'
    ALL_NAME_FILE = 'mitdb_all_categories.pkl'
    BEFORE_R_PEAK, AFTER_R_PEAK = 115, 135  # 90, 110
    TIME_LENGTH = BEFORE_R_PEAK + AFTER_R_PEAK
    INTER_POINTS_NUMBER = 2  # Number of points between each point of the ECG (P,Q,R,S,T)
    TOTAL_POINTS_NUMBER = 5 + 4 * INTER_POINTS_NUMBER  # Number of points in the ECG
    OTHER_PARAMETERS = ["previous R peak", "next R peak", "ST segment", "QT interval", "PR interval",
                        "P wave amplitude", "Q peak amplitude", "R peak amplitude", "S peak amplitude",
                        "T wave amplitude", "P wave index", "Q peak index", "R peak index", "S peak index",
                        "T wave index"]
    NUMBER_OTHER_PARAMETERS = len(OTHER_PARAMETERS)
    NEURON_NUMBER = 300  # Number of different neurons (amplitude)
    AAMI_categories = ["N", "SVEB", "VEB", "F", "Q"]
    AAMI_categories_to_supp = ["Q"]
    ALL_categories = ['NOR', 'LBBB', 'RBBB', 'AE', 'NE', 'AP', 'aAP', 'SP', 'NP', 'PVC', 'VE', 'fVN', 'U', 'fPN', 'P']
    ALL_categories_to_supp = ['P', 'fPN', 'U']
    ALL_TO_AAMI = {
        'NOR': 'N',
        'LBBB': 'N',
        'RBBB': 'N',
        'AP': 'SVEB',
        'aAP': 'SVEB',
        'NP': 'SVEB',
        'SP': 'SVEB',
        'PVC': 'VEB',
        'fVN': 'F',
        'AE': 'N',
        'NE': 'N',
        'VE': 'VEB',
    }
    KERNEL_SMOOTH_SIZE = 10
    colorscale = [[0.0, "rgb(49,54,149)"],
                  [0.1111111111111111, "rgb(69,117,180)"],
                  [0.2222222222222222, "rgb(116,173,209)"],
                  [0.3333333333333333, "rgb(171,217,233)"],
                  [0.4444444444444444, "rgb(224,243,248)"],
                  [0.5555555555555556, "rgb(254,224,144)"],
                  [0.6666666666666666, "rgb(253,174,97)"],
                  [0.7777777777777778, "rgb(244,109,67)"],
                  [0.8888888888888888, "rgb(215,48,39)"],
                  [1.0, "rgb(165,0,38)"]]
