import pandas

path = "titanic.csv"
df = pandas.read_csv(path)
total_rows = len(df.index)

all_ages = [df["Age"][i] for i in range(0, total_rows)]
average_age_total = sum(all_ages)/len(all_ages)

female_ages = [df["Age"][i] for i in range(0, total_rows) if df["Sex"][i] == "female"]
average_age_f = sum(female_ages)/len(female_ages)

male_ages = [df["Age"][i] for i in range(0, total_rows) if df["Sex"][i] == "male"]
average_age_m = sum(male_ages)/len(male_ages)

print(f'Total passengers average age: {average_age_total}\nFemale passengers average age: {average_age_f}\n'
      f'Male passengers average age: {average_age_m}')

total_fare = female_fare = male_fare = 0
for index, fare_value in enumerate(df["Fare"]):
    total_fare += fare_value
    if df["Sex"][index] == "female":
        female_fare += fare_value
    else:
        male_fare += fare_value

total_fare = total_fare / len(all_ages)
female_fare = female_fare / len(female_ages)
male_fare = male_fare / len(male_ages)

print(f'Total passengers average fare: {total_fare}\nFemale passengers average fare: {female_fare}\n'
      f'Male passengers average fare: {male_fare}')

grouped = df.groupby("Survived")

for number, frame in grouped:
    if number == 1:
        print(f'{len(frame)/len(all_ages) * 100}% of the passengers survived')

        grouped_by_sex = frame.groupby("Sex")
        for sex, fr in grouped_by_sex:
            print(f'{len(fr)/len(frame) * 100}% of the survivors were {sex}')

        grouped_by_class = frame.groupby("Pclass")
        for pclass, class_frame in grouped_by_class:
            print(f'{len(class_frame)/len(frame) * 100}% of the survivors were in class {pclass}')

        grouped_by_pc = frame.groupby("Parents/Children Aboard")
        no_pc = 0
        min_one_pc = 0
        for num_pc, pc_frame in grouped_by_pc:
            if num_pc == 0:
                no_pc += len(pc_frame)
            else:
                min_one_pc += len(pc_frame)
        print(f'{no_pc/len(frame) * 100}% of the survivors didn\'t have any children or parents aboard')
        print(f'{min_one_pc/len(frame) * 100}% of the survivors had at least one child or parent aboard')

        grouped_by_ss = frame.groupby("Siblings/Spouses Aboard")
        no_ss = 0
        min_one_ss = 0
        for num_ss, ss_frame in grouped_by_ss:
            if num_ss == 0:
                no_ss += len(ss_frame)
            else:
                min_one_ss += len(ss_frame)
        print(f'{no_ss/len(frame) * 100}% of the survivors didn\'t have any siblings or spouses aboard')
        print(f'{min_one_ss/len(frame) * 100}% of the survivors had at least one sibling or spouse sboard')

        grouped_by_age = frame.groupby("Age")
        under_18 = 0
        for age, age_frame in grouped_by_age:
            if age <= 18:
                under_18 += len(age_frame)
        print(f'{under_18/len(frame) * 100}% of the survivors were of age 18 or below')
