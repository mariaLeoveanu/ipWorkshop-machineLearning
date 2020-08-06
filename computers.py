import numpy
import urllib3
import pandas as pd
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

# http = urllib3.PoolManager()
# all_tables = []
# list_headers = []
# years = range(1993, 2021)
# months = ["06", "11"]
# for year in years:
#     if year == 2020:
#         months = ["06"]
#
#     for month in months:
#         if year == 2014 and month == "11":
#             year = 20014
#         url = f'https://www.top500.org/lists/top500/{year}/{month}/'
#         response = http.request("GET", url)
#         content = response.data.decode("utf-8")
#
#         soup = BeautifulSoup(content, "lxml")
#
#         header = soup.find_all("table")[0].find("tr")
#         print(year)
#         if not list_headers:
#             for column_name in header:
#                 if column_name != "\n":
#                     list_headers.append(column_name.text)
#             list_headers.append("Month")
#             list_headers.append("Year")
#
#         html_data = soup.find_all("table")[0].find_all("tr")[1:]
#         list_data = []
#
#         for item in html_data[:3]:
#             row = []
#             all_cells = item.find_all("td")
#             rank = int(all_cells[0].text)
#             system = all_cells[1].a.text.strip()
#             cores = int(all_cells[2].text.replace(",", "").replace(".", ""))
#             rmax = int(all_cells[3].text.replace(",", "").replace(".", ""))
#             rpeak = int(all_cells[4].text.replace(",", "").replace(".", ""))
#
#             try:
#                 power = int(all_cells[5].text.replace(",", "").replace(".", ""))
#             except:
#                 power = "NaN"
#
#             row.append(rank)
#             row.append(system)
#             row.append(cores)
#             row.append(rmax)
#             row.append(rpeak)
#             row.append(power)
#             row.append(int(month))
#             if year == 20014:
#                 row.append(2014)
#             else:
#                 row.append(year)
#
#             list_data.append(row)
#
#         all_tables.append(list_data)
#
# print(all_tables)
# flat_all_data = []
# for one_table in all_tables:
#     for one_row in one_table:
#         flat_all_data.append(one_row)
#
# dataFrame = pd.DataFrame(data=flat_all_data, columns=list_headers)
# dataFrame.set_index("Rank", inplace=True)
# dataFrame.to_csv("computers.csv")

dataFrame = pd.read_csv("computers.csv")
grouped_by_year = dataFrame.groupby("Year")
average_perfs = []
xticks = []
for year, year_frame in grouped_by_year:
    grouped_by_month = year_frame.groupby("Month")
    for month, month_frame in grouped_by_month:
        mean_performance = sum(month_frame["Rpeak (GFlop/s)"])/3
        average_perfs.append(mean_performance)
        if month == 6:
            xticks.append(f'{month}     \n{year}')
        else:
            xticks.append(f'{month}     \n\n{year}')

plt.plot(average_perfs)
x = numpy.arange(len(xticks))
plt.xticks(x, tuple(xticks))
plt.show()


perf_per_year = []
for year, year_frame in grouped_by_year:
    perf_per_year.append(sum(year_frame["Rpeak (GFlop/s)"])/len(year_frame))

previous_performance = perf_per_year[0]
avg_increase = []

for perf in perf_per_year[1:]:
    increase = perf - previous_performance
    avg_increase.append(increase)
    previous_performance = perf

bar_labels = tuple(range(1994, 2021))
y_pos = numpy.arange(len(bar_labels))
print(avg_increase)
print(len(avg_increase), len(range(1994, 2021)))
plt.bar(y_pos, avg_increase)
plt.xticks(y_pos, bar_labels)
plt.show()





