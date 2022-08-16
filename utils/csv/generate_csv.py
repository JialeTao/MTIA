import pandas as pd

fomm_csv = pd.read_csv('taichi256.csv')
fomm_csv = fomm_csv.to_dict()
source = list(fomm_csv['source'].values())
source = list(set(source))
print(source)
ll = len(source)

out_file_name = './taichi256_large.csv'
# custom_driving = ['oNkBx4CZuEg#000000#001024.mp4', 'WlDYrq8K6nk#005943#006135.mp4']
custom_driving = ['VMSqvTE90hk#007168#007312.mp4', 'VMSqvTE90hk#000165#000530.mp4','VMSqvTE90hk#000743#000887.mp4','lCb5w6n8kPs#011879#012014.mp4','FBuF0xOal9M#046824#047542.mp4','FBuF0xOal9M#013311#013440.mp4','FBuF0xOal9M#013521#013680.mp4','FBuF0xOal9M#015166#015296.mp4',
                  'FBuF0xOal9M#019516#019645.mp4', 'FBuF0xOal9M#019762#019920.mp4','FBuF0xOal9M#022427#022560.mp4','FBuF0xOal9M#025129#025327.mp4','FBuF0xOal9M#045663#046051.mp4','FBuF0xOal9M#030241#030452.mp4','FBuF0xOal9M#030929#031220.mp4','FBuF0xOal9M#032312#032523.mp4',
                  'A3ZmT97hAWU#002455#002899.mp4', 'A3ZmT97hAWU#003169#003528.mp4','A3ZmT97hAWU#007482#007618.mp4','A3ZmT97hAWU#007778#008125.mp4','aDyyTMUBoLE#000518#000884.mp4','aDyyTMUBoLE#001769#001957.mp4','DMEaUoA8EPE#000597#000810.mp4','gaccfn5JB4Y#001713#001845.mp4',
                  'L82WHgYRq6I#001384#001524.mp4', 'L82WHgYRq6I#002175#002417.mp4','L82WHgYRq6I#003345#003475.mp4','L82WHgYRq6I#003583#003821.mp4','L82WHgYRq6I#003967#004149.mp4','L82WHgYRq6I#005369#005623.mp4','OiblkvkAHWM#003461#003616.mp4','oNkBx4CZuEg#000000#001024.mp4',
                  'oNkBx4CZuEg#001024#002048.mp4', 'oNkBx4CZuEg#003647#003919.mp4','oNkBx4CZuEg#003919#004109.mp4','oNkBx4CZuEg#005811#006053.mp4','w81Tr0Dp1K8#001375#001516.mp4','w81Tr0Dp1K8#001120#001378.mp4','w81Tr0Dp1K8#015670#015801.mp4','w81Tr0Dp1K8#008965#009131.mp4',
                  'w81Tr0Dp1K8#012900#013080.mp4', 'WlDYrq8K6nk#002253#002686.mp4','WlDYrq8K6nk#004875#005151.mp4','WlDYrq8K6nk#005943#006135.mp4','WlDYrq8K6nk#007058#007241.mp4','ZFpTP2fSThw#000744#000945.mp4','ZFpTP2fSThw#005265#005400.mp4']
custom_csv = open(out_file_name, 'w')

custom_csv = {'distance':[],'source':[],'driving':[],'frame':[]}
for dri in custom_driving:
    custom_csv['distance']+=[0]*ll
    custom_csv['frame']+=[0]*ll
    custom_csv['source']+=source
    custom_csv['driving']+=[dri]*ll
print(custom_csv)
df = pd.DataFrame(custom_csv)
# print(df)
df.to_csv(out_file_name, index=None)
