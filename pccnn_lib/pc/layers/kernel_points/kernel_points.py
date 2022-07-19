import numpy as np

kernel_pts_dict = {
    '2_8' : np.array([  [0.000000,0.000000],
                        [-0.150918,-0.988547],
                        [-0.292942,0.956131],
                        [0.997345,0.072839],
                        [0.678780,-0.734340],
                        [-0.866972,-0.498356],
                        [0.564886,0.825168],
                        [-0.930179,0.367106]]),
    '2_16' : np.array([ [0.000000,0.000000],
                        [-1.237629,0.078798],
                        [0.422437,1.064617],
                        [0.473192,-0.392059],
                        [-0.307508,1.201405],
                        [-0.613267,0.039047],
                        [-0.457389,-1.152705],
                        [1.047579,0.663712],
                        [-0.967525,-0.612992],
                        [1.143052,-0.072776],
                        [-0.226646,-0.571185],
                        [0.954947,-0.791209],
                        [0.284008,-1.109596],
                        [-0.152374,0.595318],
                        [-0.881971,0.730746],
                        [0.519095,0.328880]]),
    '2_32' : np.array([ [0.000000,0.000000],
                        [1.119605,-0.630469],
                        [-1.262018,0.371594],
                        [-0.015669,1.284845],
                        [-0.014040,0.798845],
                        [0.438287,-0.053057],
                        [-0.794972,-0.088981],
                        [-0.361705,-0.238225],
                        [0.191212,0.388862],
                        [-0.536309,1.202488],
                        [0.717505,0.345122],
                        [-0.777990,0.361163],
                        [0.603625,-0.530310],
                        [0.235573,-0.854319],
                        [1.211237,0.404782],
                        [-0.733906,-1.074511],
                        [-0.249899,-0.755970],
                        [0.772645,-1.026402],
                        [-0.979553,0.857064],
                        [-1.276163,-0.155422],
                        [-0.681322,-0.565943],
                        [1.338969,-0.116467],
                        [-0.240833,-1.254128],
                        [-0.458921,0.725747],
                        [0.968551,0.870143],
                        [0.877790,-0.123455],
                        [0.293250,-1.311861],
                        [-1.144393,-0.665235],
                        [0.473675,0.749268],
                        [0.506771,1.222253],
                        [0.108104,-0.427873],
                        [-0.323165,0.285035]]),
    '2_64' : np.array([ [0.000000,0.000000],
                        [1.167834,-0.720430],
                        [0.217590,1.033306],
                        [-0.870858,-0.503692],
                        [1.367745,0.013927],
                        [0.295680,0.156715],
                        [-0.915792,0.438330],
                        [0.567936,0.404440],
                        [-1.019136,0.115124],
                        [0.789214,0.677436],
                        [-1.064014,-0.225638],
                        [1.368556,-0.373371],
                        [-0.323051,0.224182],
                        [-0.033591,-0.575621],
                        [0.574359,-0.846828],
                        [-0.818891,-1.111246],
                        [-0.523721,-0.394251],
                        [-0.059016,-0.921163],
                        [0.475694,-0.463947],
                        [1.001906,0.032024],
                        [-0.483579,-1.338396],
                        [-0.182586,-0.286533],
                        [-1.200121,0.675119],
                        [-1.304184,-0.508480],
                        [0.945899,1.061782],
                        [-0.034178,0.310887],
                        [0.615761,-1.235637],
                        [-0.338526,-0.647921],
                        [-1.339107,0.317020],
                        [-0.663391,0.173749],
                        [0.708595,-0.225819],
                        [-1.087011,-0.832365],
                        [-0.755237,0.752388],
                        [-0.728763,-0.144306],
                        [0.597302,0.997003],
                        [0.929322,-1.011063],
                        [1.316983,0.379066],
                        [0.411531,0.720957],
                        [1.022850,-0.320817],
                        [-0.546522,0.485510],
                        [-0.991993,1.015626],
                        [-1.402306,-0.077330],
                        [-0.388330,-0.068522],
                        [0.486135,1.336684],
                        [-0.672943,-0.765941],
                        [0.937347,0.363902],
                        [-0.397668,-0.989175],
                        [-0.110540,-1.297676],
                        [0.071053,1.381912],
                        [-0.428196,0.861365],
                        [0.257478,-0.690175],
                        [0.657388,0.102404],
                        [0.227469,0.465136],
                        [-0.145957,1.031449],
                        [-0.642190,1.173646],
                        [0.035780,0.735333],
                        [0.147829,-0.307593],
                        [-0.324325,1.355639],
                        [-0.235277,0.578732],
                        [0.262909,-1.036150],
                        [0.241518,-1.398640],
                        [0.804627,-0.600199],
                        [1.167242,0.723153],
                        [0.390005,-0.127947]]),
    '3_8' : np.array([  [0.000000,0.000000,0.000000],
                        [0.583069,0.017557,-0.802703],
                        [0.593079,0.550196,0.593251],
                        [-0.147796,0.966117,-0.226858],
                        [0.678900,-0.618589,0.402835],
                        [-0.582526,-0.050005,0.801732],
                        [-0.860718,0.065902,-0.510825],
                        [-0.263991,-0.932610,-0.257466]]),
    '3_16' : np.array([ [0.000000,0.000000,0.000000],
                        [-0.362877,0.789640,0.496690],
                        [-0.665216,-0.745738,-0.057587],
                        [0.317646,0.930844,-0.185878],
                        [0.166266,0.358249,-0.911110],
                        [0.745918,-0.397344,0.521361],
                        [0.728800,-0.371354,-0.579654],
                        [-0.166266,-0.715482,0.682265],
                        [-0.554686,0.746530,-0.374261],
                        [-0.139258,0.057401,0.991137],
                        [-0.912333,0.038953,0.390155],
                        [-0.130192,-0.527205,-0.840853],
                        [-0.815681,-0.016221,-0.582607],
                        [0.285401,-0.956719,-0.071690],
                        [0.555437,0.509367,0.658781],
                        [0.947041,0.299078,-0.136742]]),
    '3_32' : np.array([ [0.000000,0.000000,0.000000],
                        [1.002142,0.123266,-0.173247],
                        [0.178418,0.983773,0.214982],
                        [-0.465128,0.242646,0.239527],
                        [0.671389,-0.416119,-0.644589],
                        [0.281187,-0.172210,1.020520],
                        [-0.051716,-0.563490,-0.093961],
                        [0.039090,-0.712240,-0.788008],
                        [0.683730,0.265639,-0.772334],
                        [-0.791833,-0.597833,0.281860],
                        [-0.321955,-0.494658,0.822614],
                        [0.584585,0.782684,-0.296272],
                        [-0.455063,0.826405,0.544502],
                        [-0.351344,0.945743,-0.212822],
                        [0.072710,-0.043075,-1.100980],
                        [0.831945,0.618604,0.347001],
                        [0.398019,-1.012333,-0.183440],
                        [-0.948664,0.542200,-0.076968],
                        [-0.645913,0.420874,-0.733403],
                        [-0.000186,0.213868,-0.527588],
                        [0.242179,0.608709,0.856743],
                        [-0.186569,-1.019959,0.360000],
                        [0.927387,-0.134109,0.593242],
                        [-0.982306,-0.160168,-0.255799],
                        [0.386109,0.131234,0.398410],
                        [-0.355481,0.199636,0.972692],
                        [0.427929,-0.728809,0.582776],
                        [0.049793,0.795107,-0.773183],
                        [0.868628,-0.535291,0.012269],
                        [-0.952980,0.013645,0.567056],
                        [-0.587546,-0.843471,-0.364764],
                        [-0.578137,-0.272688,-0.815454]]),
    '3_64' : np.array([ [0.000000,0.000000,0.000000],
                        [-0.782221,0.784071,-0.176035],
                        [0.881264,-0.660647,0.106887],
                        [-0.207994,-0.735639,0.807352],
                        [-0.911389,0.235787,-0.610989],
                        [-1.060309,0.307614,-0.027923],
                        [0.724755,0.823996,-0.328133],
                        [-0.081036,0.598705,0.107234],
                        [-0.373104,-0.336004,0.400089],
                        [-1.095043,-0.226656,-0.229394],
                        [-0.189210,-0.601488,-0.067518],
                        [0.607822,-0.084645,0.113013],
                        [-0.827556,0.639793,0.434305],
                        [1.117058,-0.104099,0.221914],
                        [-0.407125,1.024001,0.221152],
                        [0.407117,0.629416,0.846777],
                        [-1.009099,0.027359,0.461792],
                        [0.024934,-1.015582,0.391723],
                        [0.198439,0.013869,-1.077351],
                        [-0.252435,1.008815,-0.340401],
                        [-0.188657,-0.200751,1.079485],
                        [-0.763660,-0.336306,-0.719899],
                        [0.297913,-0.541265,-0.900237],
                        [-0.332880,0.720337,0.736505],
                        [0.870816,0.277370,0.649534],
                        [0.336879,-0.518616,-0.157881],
                        [0.213298,0.894501,-0.620086],
                        [0.298519,-0.505334,0.957508],
                        [0.660653,0.840929,0.314315],
                        [0.370366,0.330664,0.402403],
                        [-0.629688,0.220602,0.888783],
                        [-0.513728,0.686969,-0.729371],
                        [-0.562722,-0.901686,0.349263],
                        [0.745858,-0.147221,-0.856301],
                        [0.449995,0.032588,0.996262],
                        [0.218320,-0.391054,0.422299],
                        [-0.582293,-0.191509,-0.147746],
                        [1.004932,0.446192,0.092218],
                        [0.413500,-1.046329,-0.039813],
                        [0.000102,0.357580,-0.517199],
                        [0.541690,-0.831907,0.523719],
                        [-0.475658,0.335524,-0.277495],
                        [-0.025716,0.521490,-1.016208],
                        [0.711029,-0.713693,-0.508504],
                        [0.803556,-0.319966,0.670946],
                        [1.017100,-0.268298,-0.324961],
                        [-0.952559,-0.497484,0.267588],
                        [0.138364,-0.937660,-0.562818],
                        [-0.701827,-0.370867,0.811234],
                        [-0.769377,-0.769183,-0.236391],
                        [0.971207,0.273123,-0.443149],
                        [-0.236958,-0.385720,-1.046424],
                        [0.111323,0.975457,0.514329],
                        [-0.149636,-0.229004,-0.544845],
                        [-0.053980,0.095772,0.617291],
                        [-0.499569,0.242926,0.297968],
                        [-0.390425,-0.789756,-0.661453],
                        [0.411856,-0.057104,-0.472478],
                        [0.425521,0.445321,-0.154512],
                        [-0.236868,-1.111137,-0.135819],
                        [0.569706,0.451126,-0.832356],
                        [0.239661,1.080648,-0.042854],
                        [-0.466258,0.130017,-0.970723],
                        [-0.053118,0.373936,1.072296]]),
    '4_8' : np.array([  [0.000000,0.000000,0.000000,0.000000],
                        [-0.800436,0.114746,0.552907,-0.211356],
                        [-0.519328,0.014862,-0.832523,-0.203039],
                        [-0.055131,-0.979273,0.058395,-0.170095],
                        [0.519328,-0.014858,0.832523,0.203041],
                        [-0.232069,0.360814,-0.067712,0.897625],
                        [0.287200,0.618451,0.009318,-0.727534],
                        [0.800436,-0.114742,-0.552907,0.211358]]),
    '4_16' : np.array([ [0.000000,0.000000,0.000000,0.000000],
                        [-0.322824,0.332314,0.682333,0.565541],
                        [0.595545,0.544064,0.030458,0.589898],
                        [-0.868683,-0.365594,0.005995,0.334139],
                        [-0.500813,0.774175,-0.293508,0.252616],
                        [0.481770,0.098055,0.853065,-0.175180],
                        [-0.080437,-0.079624,-0.629792,0.768515],
                        [-0.621760,0.287207,0.506880,-0.523186],
                        [0.250073,-0.199472,-0.037626,-0.946684],
                        [0.283752,0.830264,0.070721,-0.474561],
                        [-0.203565,-0.749286,0.557472,-0.293948],
                        [-0.613186,0.003243,-0.599236,-0.514741],
                        [0.245605,-0.572335,0.331699,0.708667],
                        [0.017375,-0.839349,-0.537793,-0.075459],
                        [0.400225,0.272644,-0.859205,-0.165462],
                        [0.936909,-0.336319,-0.081465,-0.050169]]),
    '4_32' : np.array([ [0.000000,0.000000,0.000000,0.000000],
                        [0.686036,0.507679,-0.393381,0.349591],
                        [-0.231999,-0.140760,-0.637383,0.720235],
                        [0.292841,-0.321947,-0.524592,-0.731401],
                        [0.356304,-0.769276,-0.244148,0.472010],
                        [-0.602875,0.753652,-0.097857,-0.237532],
                        [-0.144017,0.648977,-0.695097,0.276570],
                        [0.629492,-0.137312,0.282598,-0.711846],
                        [-0.714812,0.146003,0.523951,-0.442119],
                        [-0.898134,0.104117,-0.369483,0.210097],
                        [0.422365,0.574634,-0.445629,-0.532822],
                        [0.485002,-0.037998,-0.134529,0.864491],
                        [-0.268920,0.230587,-0.147180,-0.924684],
                        [-0.066369,-0.321353,0.503088,0.798840],
                        [-0.233522,-0.657726,-0.712473,-0.093695],
                        [0.112252,0.621578,0.476601,-0.609904],
                        [0.140953,-0.005780,0.976194,-0.150765],
                        [-0.404401,0.162239,-0.809679,-0.394703],
                        [-0.135788,-0.416417,0.457677,-0.777387],
                        [0.840780,0.406206,0.342692,-0.117541],
                        [-0.330447,0.474159,0.003963,0.812017],
                        [-0.739350,-0.480892,-0.122599,-0.449375],
                        [0.153144,0.978120,0.077649,0.127627],
                        [0.217234,-0.918718,0.016065,-0.327765],
                        [0.423890,0.364749,0.593967,0.580416],
                        [-0.531430,-0.718408,-0.083846,0.444135],
                        [-0.342777,0.585884,0.700543,0.227183],
                        [-0.259957,-0.705929,0.657853,0.024660],
                        [-0.806149,-0.074388,0.451632,0.380064],
                        [0.915495,-0.287633,-0.270257,-0.064805],
                        [0.654820,-0.485099,0.541075,0.194294],
                        [0.380402,-0.078846,-0.917562,0.083899]]),
    '4_64' : np.array([ [0.000000,0.000000,0.000000,0.000000],
                        [-0.014144,0.129904,0.249939,-0.959236],
                        [-0.869066,-0.370181,0.323624,-0.075225],
                        [0.254645,-0.464972,-0.710577,-0.461642],
                        [0.507588,-0.513113,-0.095554,-0.686542],
                        [-0.170724,-0.543693,-0.315799,0.759204],
                        [0.631761,-0.759908,-0.139376,-0.058221],
                        [0.266819,0.710665,-0.605340,-0.244238],
                        [0.644029,0.082630,-0.620971,-0.440837],
                        [0.628628,0.351795,-0.630721,0.289262],
                        [-0.426157,0.746252,0.200334,-0.473048],
                        [-0.027767,0.391597,0.907723,0.148522],
                        [0.476711,-0.095005,0.873721,0.030532],
                        [-0.831335,-0.331266,-0.378233,0.231925],
                        [0.583500,-0.469498,-0.266952,0.607779],
                        [-0.597425,0.688753,-0.161073,0.375531],
                        [-0.568139,-0.055561,0.794907,0.212302],
                        [0.318024,-0.300568,0.569244,-0.696568],
                        [0.315535,0.821868,0.469476,-0.046412],
                        [0.043201,0.165730,-0.978109,-0.103915],
                        [0.873272,-0.270728,0.319047,-0.255170],
                        [-0.023654,0.455070,-0.114695,0.882921],
                        [0.637997,0.145342,-0.008336,-0.756987],
                        [-0.324231,-0.744534,0.584627,-0.001890],
                        [0.142978,-0.845270,0.230236,0.458443],
                        [-0.643828,0.262641,-0.699237,0.171989],
                        [-0.880699,0.134717,0.169522,0.425459],
                        [-0.086594,0.991528,-0.109045,0.002814],
                        [-0.426484,0.654902,-0.513378,-0.353706],
                        [0.874859,0.296991,0.311353,0.223951],
                        [-0.743458,-0.419536,-0.179593,-0.490204],
                        [-0.118472,-0.382699,-0.267700,-0.876585],
                        [-0.471595,-0.019069,-0.746615,-0.468745],
                        [0.111639,-0.849717,-0.476802,0.204880],
                        [-0.071947,0.610816,-0.670366,0.410591],
                        [0.613252,0.344219,0.583742,-0.408595],
                        [0.029765,-0.356893,0.784489,0.503382],
                        [-0.490345,-0.856420,-0.073973,0.149391],
                        [-0.663820,0.105877,0.537397,-0.506886],
                        [-0.929801,0.279808,-0.115301,-0.198786],
                        [0.448983,0.782023,-0.098700,0.420222],
                        [0.350137,-0.748524,0.541621,-0.149740],
                        [-0.114682,0.774309,0.376183,0.500909],
                        [0.072620,0.262884,-0.521612,-0.806199],
                        [-0.141198,-0.221139,0.914534,-0.309780],
                        [0.641682,0.176013,-0.081348,0.743739],
                        [-0.268705,-0.289083,-0.853246,0.339897],
                        [-0.049385,0.437747,0.721787,-0.533291],
                        [-0.031183,-0.913493,-0.012553,-0.399139],
                        [-0.525542,0.053941,-0.321488,0.784932],
                        [-0.340277,0.190650,0.485763,0.782337],
                        [-0.355482,-0.640298,-0.643837,-0.232181],
                        [0.392413,0.307186,0.598890,0.624177],
                        [-0.612997,0.594988,0.515450,0.054629],
                        [0.769618,0.592359,-0.113831,-0.209770],
                        [0.211389,0.726147,0.021453,-0.652426],
                        [-0.557859,0.205009,-0.117301,-0.797766],
                        [0.468940,-0.320898,-0.805182,0.174794],
                        [-0.348210,-0.509649,0.384866,-0.684132],
                        [0.688895,-0.389031,0.408166,0.453504],
                        [0.140012,-0.217652,0.204981,0.942809],
                        [-0.529194,-0.471690,0.282028,0.644947],
                        [0.951169,-0.128249,-0.273077,0.036517],
                        [0.164329,0.023929,-0.645128,0.745715]]),
}