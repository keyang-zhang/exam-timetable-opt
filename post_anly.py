from new import GAOptimizer
import pandas as pd
from prettytable import PrettyTable

KPI_CONSEC_1 = "2 consecutive exams in a week"
KPI_CONSEC_2 = "3 consecutive exams in a week"
KPI_OVERLOAD_1 = "more than 1 exam in a day"
KPI_OVERLOAD_2 = "more than 3 exams in a week"
KPI_OVERLOAD_3 = "4 exams in a week"
KPI_OVERLOAD_4 = "5 exams in a week"
KPI_SET = (KPI_CONSEC_1, KPI_CONSEC_2, KPI_OVERLOAD_1, KPI_OVERLOAD_2, KPI_OVERLOAD_3, KPI_OVERLOAD_4)

regis_file = "./data/exam_registration.csv"
timespace_file = "./data/exam_timetable_2023.csv"
rooms = ("R060", "R064", "R301", "R307", "R315")
opt = GAOptimizer()
opt.process_register_data(regis_file, "CID")
opt.process_spatio_time_data(timespace_file, rooms)

best_table = pd.read_csv("./data/optimized_table_best.csv")

ptable = PrettyTable()
dtable = {}
ptable.field_names = ["Module", "Date", "Slot", "Room"]
for _, row in best_table.iterrows():
    label = row["weekday"]
    slot = row["slot"]
    week = row["week"]
    day = row["day"]
    for room in rooms:
        if not pd.isna(row[room]):
            exam = row[room]
            ptable.add_row([exam, label, slot, room])
            dtable[exam] = (day, slot, room)

# with open("best_table.txt", "w") as text_file:
#     print(ptable.get_string(), file=text_file)

opt.exam2spats = dtable
student_specific_table = {"student id": []}
student_specific_table.update({kpi: [] for kpi in KPI_SET})
student_specific_table["exams"] = []
# calculate KPI values - 2/3 consecutive exams and 3 exams in a week
for stu_id, registered_exams in opt.student2exams.items():
    kpi_value = {kpi: 0 for kpi in KPI_SET}
    student_exam_spatss = [opt.exam2spats[exam] for exam in registered_exams]
    student_exam_spatss.sort(key=lambda x: x[0])

    # check whether this student has 2 or 3 consecutive exams
    consecutive_count = 1
    for pre_exam, next_exam in zip(student_exam_spatss[:-1], student_exam_spatss[1:]):
        if next_exam[0] == pre_exam[0]:
            kpi_value[KPI_OVERLOAD_1] += 1
        elif next_exam[0] - pre_exam[0] == 1:
            consecutive_count += 1
        else:
            if consecutive_count == 2:
                kpi_value[KPI_CONSEC_1] += 1
            elif consecutive_count == 3:
                kpi_value[KPI_CONSEC_2] += 1
            consecutive_count = 1

    # check whether this student has more than 3 exams in a week
    student_exam_dates_set = set(spats[0] for spats in student_exam_spatss)
    for week in opt.week2date_dict:
        if len(student_exam_dates_set.intersection(set(opt.week2date_dict[week]))) > 3:
            kpi_value[KPI_OVERLOAD_2] += 1
        if len(student_exam_dates_set.intersection(set(opt.week2date_dict[week]))) == 4:
            kpi_value[KPI_OVERLOAD_3] += 1
        if len(student_exam_dates_set.intersection(set(opt.week2date_dict[week]))) == 5:
            kpi_value[KPI_OVERLOAD_4] += 1

    student_specific_table["student id"].append(stu_id)
    for kpi in KPI_SET:
        student_specific_table[kpi].append(kpi_value[kpi])

    student_specific_table["exams"].append(str(student_exam_spatss))

pd.DataFrame(student_specific_table).to_csv("student_table.csv", index=False)
