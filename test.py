import csv
from pandas import read_csv
from pandas import DataFrame
from pandas import Series
from matplotlib import pyplot
import numpy as np
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMA
from collections import defaultdict

columns = defaultdict(list)

values = []

data = read_csv("test_data.csv")

#В первую очередь необходимо построить гисторграмму, которая позволит провести
#визуальный анализ данных
pyplot.hist(data ["counter_data"], bins= int(200))
pyplot.title ("Database with anomalies")
pyplot.show ()

#Далее выявим аномальные точки с использованием методом стандартного отклонения.
#Выбор данного метода обусловлен тем, что модель должна выполняться быстро
#и в распоряжении исследователя находится временной ряд.

anomalies = []


def find_anomalies(data):

    data_std = np.std(data)
    data_mean = np.mean(data)
    sigma = data_std * 3

    lower_limit  = data_mean - sigma
    upper_limit = data_mean + sigma

    for outlier in data:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    print ("The following anomalies were found in the database", anomalies)

find_anomalies(data ["counter_data"])

#Найденные в выборке аномальные значения необходимо удалить. Для того, чтобы
#не менять исходные данные наилучшим способом будет сформировать новую строку
#в программе и удалить аномалии из данной строки. Удаление аномалий необходимо
#для качественного прогнозирования в дальнейшем.

print ("Removing anomalies")

with open("test_data.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        for (k,v) in row.items():
            columns[k].append(v)

values = columns['counter_data']


while '' in values:
    values.remove('')

for value in values:
    for anomaly in anomalies:
        if float(value) == anomaly:
            values.remove (value)


print ("Database without anomalies", values)

values = [float(x) for x in values]

value_for_predickt = values [:1000]



#После удаление аномалий из базы данных необходимо утвердить определенную модель
#для дальнейшего прогнозирования. Для этого необходимо определить стационарность
#данного временного ряда. С этой целью воспользуемся тестом Дикки-Фуллера

stationary = 0
test = sm.tsa.adfuller(values)
print ("adf: ", test[0])
print ("p-value: " , test[1])
print("Critical values: ", test[4])
if test[0]> test[4]['5%']:
    print ("The row is not stationary")
else:
    print ("The row is stationary")
    stationary = 1


# В случае если ряд нестационарный, то существует возможность того, что ряд
# имеет сезонные изменения. Для проверки данного факта стоит провести тест
# по автокорреляционной матрице. При стационарности ряд не проверяется на
#сезонность по определению стационарности.

if stationary == 0:
     series = Series(values)
     print (series)
     plot_acf(series)
     pyplot.show()


# В данном примере ряд обозначен как стационарный, однако для того, чтобы модель
# могла работать для других рядов стоит прописать как минимум 2 метода прогнозирования
# для стационаных рядов и нестационарных рядов. Для первых стоит использовать ARMA
# модель, тогда как для вторых - ARIMA.

if stationary == 1:
     model = ARMA(value_for_predickt, order=(1, 2))
     model_fit = model.fit()

     yhat = model_fit.predict(1, 6)
     print(yhat)

if stationary == 0:
    model = ARIMA(value_for_predickt, order=(1, 2))
    model_fit = model.fit()

    yhat = model_fit.predict(1, 6)
    print(yhat)

# В ходе выполнения данного скрипта становится возможным выделить прогнозные
#значения переменных и модель для данного временного ряда. Отдельно интересно
# отобразить данные по прогнозным значениям и по модели в целом.

pyplot.hist(yhat, bins= int(10))
pyplot.title ("Predict")
pyplot.show ()

pyplot.plot (values)
pyplot.title ("Data without anomaly")
pyplot.show ()

#Последний пункт: запись результатов исследования в csv файл

anomalies = ["anomalies"] + anomalies

with open('Test.csv', 'w') as f:
    f = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    f.writerow(data ["COUNTER_ID"])
    f.writerow(data ["counter_data"])
    f.writerow(anomalies)
    f.writerow (yhat)
