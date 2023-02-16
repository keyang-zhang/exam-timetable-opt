import csv
import random
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from deap import algorithms
from deap import base
from deap import creator
from deap import tools

INF = 1e7
NO_EXAM_PLACEHOLDER = "no exam"
KPI_CONSEC_1 = "2 consecutive exams in a week"
KPI_CONSEC_2 = "3 consecutive exams in a week"
KPI_OVERLOAD_1 = "more than 1 exam in a day"
KPI_OVERLOAD_2 = "more than 3 exams in a week"
KPI_OVERLOAD_3 = "4 exams in a week"
KPI_OVERLOAD_4 = "5 exams in a week"
KPI_EXAM_DURA = "duration of the exam period"
KPI_SET = (KPI_CONSEC_1, KPI_CONSEC_2, KPI_OVERLOAD_1, KPI_OVERLOAD_2, KPI_EXAM_DURA, KPI_OVERLOAD_3, KPI_OVERLOAD_4)


class GAOptimizer:
    def __init__(self):
        self.available_spatio_timeslots = None
        self.week2date_dict = None  # {week:[days]}
        self.room_caps = dict()
        # processed data
        self.exam2students = None  # {exam_code: set(students)}
        self.student2exams = None  # {student_id: set(selected_exams)}
        self.no_conflict_exams_pairs = None  # [(exam1,exam2),...]
        self.conflict_exams_pairs = None  # [(exam1,exam2),...]
        # all exams are classified into three categories - bound, fixed and arranged
        self.bindings = dict()  # the "key" exam is bound with the "value" exam
        self.fixed_exams = dict()  # key is exam, value is date
        self.arranged_exams = list()  # a sequence of exams (including placeholders) that have been arranged
        # ga_results
        self.ga_pop = None
        self.ga_log = None
        self.exam2spats = None  # optimised and complete exam timetable

    def initialize(self, spatime_file, rooms, room_caps, fixed_exams, regis_datafile,
                   id_col="ID", day_col="day", week_col="week", slot_col="slot"):
        self.fixed_exams = fixed_exams
        self.room_caps = room_caps

        self.process_spatio_time_data(spatime_file, rooms, day_col, week_col, slot_col)
        self.process_register_data(regis_datafile, id_col)
        self.check_conflict()

    def process_spatio_time_data(self, spatime_file, rooms, day_col="day", week_col="week", slot_col="slot"):
        spatime_table = pd.read_csv(spatime_file)
        week2date_dict = {}
        available_spatio_timeslots = []
        for _, row in spatime_table.iterrows():
            day = row[day_col]
            week = row[week_col]
            for room in rooms:
                if pd.isna(row[room]):
                    available_spatio_timeslots.append((day, row[slot_col], room))
            if week2date_dict.get(week) is None:
                week2date_dict[week] = []
            else:
                if day not in week2date_dict[week]:
                    week2date_dict[week].append(day)

        self.week2date_dict = week2date_dict
        self.available_spatio_timeslots = available_spatio_timeslots
        self.ts_table = spatime_table

    def process_register_data(self, regis_datafile, id_column="ID"):
        student_regis_table = pd.read_csv(regis_datafile, index_col=id_column)

        students = {}
        for student_id, row in student_regis_table.iterrows():
            selected_exams = []
            for exam_code, selected in row.items():
                if selected:
                    selected_exams.append(exam_code)
            students[student_id] = set(selected_exams)

        exams = {}
        for exam_code, col in student_regis_table.iteritems():
            registered_students = []
            for student_id, registered in col.items():
                if registered:
                    registered_students.append(student_id)

            # only arrange the exams for which at least one student registered
            if len(registered_students):
                exams[exam_code] = set(registered_students)

        self.student2exams = students
        self.exam2students = exams
        return students, exams

    def check_conflict(self):
        no_overlap_exams_pairs = []
        overlap_exams_pairs = []
        for i, j in combinations(self.exam2students.keys(), 2):
            if not set.intersection(self.exam2students[i], self.exam2students[j]):
                no_overlap_exams_pairs.append((i, j))
            else:
                overlap_exams_pairs.append((i, j))

        self.no_conflict_exams_pairs = no_overlap_exams_pairs
        self.conflict_exams_pairs = overlap_exams_pairs

    def generate_bindings(self):
        """ Compress the exams - put 2 non-conflicting exams on the same day"""
        bindings = {}  # the "key" exam will follow the "value" exam
        free_exams = set(self.exam2students.keys())  # exams have not been bound with others
        for exam_1, exam_2 in self.no_conflict_exams_pairs:
            if exam_1 in free_exams and exam_2 in free_exams:
                if exam_1 in self.fixed_exams:  # the "key" exam cannot be a fixed exam
                    bindings[exam_2] = exam_1
                else:
                    bindings[exam_1] = exam_2
                free_exams.remove(exam_1)
                free_exams.remove(exam_2)

        self.bindings = bindings
        return bindings

    def optimize(self, kpi_coef, pop_size=100, crossover_rate=0, mutation_rate=0.5, num_generation=200):

        free_exams = list(set(self.exam2students.keys()) - set(self.fixed_exams.keys()) - set(self.bindings.keys()))
        if len(free_exams) > len(self.available_spatio_timeslots):
            raise ValueError("the number of exams exceeds the number of available spaces")
        else:
            num_no_exam = len(self.available_spatio_timeslots) - len(free_exams)
            free_exams += [NO_EXAM_PLACEHOLDER] * num_no_exam

        # define chromosome and individual
        creator.create("Fitness", base.Fitness, weights=(1,))
        creator.create("Individual", list, fitness=creator.Fitness)
        toolbox = base.Toolbox()
        toolbox.register("chromosome", random.sample, free_exams, len(free_exams))
        toolbox.register("individual", tools.initIterate, creator.Individual,
                         toolbox.chromosome)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # define evaluation, mutation, crossover and selection methods
        toolbox.register("evaluate", GAOptimizer.evaluate,
                         fixed_exams=self.fixed_exams,
                         bindings=self.bindings,
                         available_spatio_timeslots=self.available_spatio_timeslots,
                         student2exams=self.student2exams,
                         exam2students=self.exam2students,
                         week2date_dict=self.week2date_dict,
                         kpi_coef=kpi_coef,
                         room_caps=self.room_caps,
                         conflict_exams_pairs=self.conflict_exams_pairs)
        toolbox.register("mate", tools.cxPartialyMatched)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=5)

        # create population and start evolving
        pop = toolbox.population(n=pop_size)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(key=lambda ind: ind.fitness.values)
        stats.register("best", max)
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=crossover_rate, mutpb=mutation_rate, ngen=num_generation,
                                       stats=stats, halloffame=hof,
                                       verbose=True)
        self.ga_pop = pop
        self.ga_log = log
        self.arranged_exams = hof[0]

        exam2spats = self.gen_full_table(self.available_spatio_timeslots,
                                         self.arranged_exams,
                                         self.fixed_exams,
                                         self.bindings)

        self.exam2spats = exam2spats
        return exam2spats

    def get_kpis(self):
        return self.calculate_kpis(self.exam2spats, self.student2exams, self.week2date_dict)

    def get_feasibility(self):
        feasible = True
        cap_feasible = True
        time_feasible = True
        # check feasibility - capacity
        for exam, spats in self.exam2spats.items():
            if exam not in self.fixed_exams:
                stu_n = len(self.exam2students[exam])
                cap = room_caps[spats[2]]
                if stu_n > cap:
                    feasible = False
                    cap_feasible = False
                    print(
                        f"exam {exam} is arrange to {spats} but capacity is not enough. student: {stu_n} while capacity: {cap}")

        # check feasibility - conflict exams
        for e1, e2 in self.conflict_exams_pairs:
            spats1, spats2 = self.exam2spats[e1], self.exam2spats[e2]
            if spats1[0] == spats2[0] and spats1[1] == spats2[1]:
                feasible = False
                time_feasible = False
                print(f"{e1} and {e2} are arranged on the same day and slot, but they are conflicting")
        return feasible, cap_feasible, time_feasible

    def output_table(self, path="./data/optimized_table.csv"):
        table = self.ts_table.fillna("")
        table = table.astype(str)
        for exam, spats in self.exam2spats.items():
            row_idx = table.query(f'day == "{spats[0]}" & slot == "{spats[1]}"').index[0]
            table.at[row_idx, spats[2]] = exam
        table.to_csv(path, index=False)

    @staticmethod
    def evaluate(individual, fixed_exams: dict, bindings: dict, available_spatio_timeslots: list,
                 student2exams: dict, exam2students: dict, week2date_dict: dict, kpi_coef: dict,
                 room_caps: dict, conflict_exams_pairs):
        # combine three types of exams to get complete exam timetable
        exam2spats = GAOptimizer.gen_full_table(available_spatio_timeslots, individual, fixed_exams, bindings)

        kpi_value = GAOptimizer.calculate_kpis(exam2spats, student2exams, week2date_dict)

        # calculate the weighted fitness
        fitness = sum(kpi_coef[kpi] * kpi_value[kpi] for kpi in KPI_SET)

        # penalize capacity feasibility violation
        for exam, spats in exam2spats.items():
            if exam not in fixed_exams:
                stu_n = len(exam2students[exam])
                cap = room_caps[spats[2]]
                if stu_n > cap:
                    fitness -= INF

        # penalize conflicting-exam feasibility violation
        for e1, e2 in conflict_exams_pairs:
            spats1, spats2 = exam2spats[e1], exam2spats[e2]
            if spats1[0] == spats2[0] and spats1[1] == spats2[1]:
                fitness -= INF

        return fitness,

    @staticmethod
    def calculate_kpis(exam2spats, student2exams, week2date_dict):
        # init kpi values
        kpi_value = {kpi: 0 for kpi in KPI_SET}
        # calculate KPI values - 2/3 consecutive exams and 3 exams in a week
        for _, registered_exams in student2exams.items():
            student_exam_spatss = [exam2spats[exam] for exam in registered_exams]
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
            for week in week2date_dict:
                if len(student_exam_dates_set.intersection(set(week2date_dict[week]))) > 3:
                    kpi_value[KPI_OVERLOAD_2] += 1
                if len(student_exam_dates_set.intersection(set(week2date_dict[week]))) == 4:
                    kpi_value[KPI_OVERLOAD_3] += 1
                if len(student_exam_dates_set.intersection(set(week2date_dict[week]))) == 5:
                    kpi_value[KPI_OVERLOAD_4] += 1
        # calculate kpi value - the duration of the exam period (captured by the day of last exams
        last_exam_date = 0
        for spats in exam2spats.values():
            if spats[0] > last_exam_date:
                last_exam_date = spats[0]
        kpi_value[KPI_EXAM_DURA] = last_exam_date
        return kpi_value

    @staticmethod
    def gen_full_table(available_spatio_timeslots, arranged_exams, fixed_exams, bindings):
        """combine three types of exams to get complete exam timetable"""
        exam2spats = {exam: spats for exam, spats in zip(arranged_exams, available_spatio_timeslots) if
                      exam != NO_EXAM_PLACEHOLDER}
        exam2spats.update(fixed_exams)
        exam2spats.update({exam_k: exam2spats[exam_v] for exam_k, exam_v in bindings.items()})
        return exam2spats


if __name__ == "__main__":
    # input
    rooms = ("R060", "R064", "R301", "R307", "R315")
    room_caps = {"R060": 84, "R064": 63, "R301": 68, "R307": 63, "R315": 28}
    fixed_exams = {"CIVE97122": (5, "am", "R315")}
    kpi_coef = {KPI_CONSEC_1: -0,
                KPI_CONSEC_2: -0,
                KPI_OVERLOAD_1: -0,
                KPI_OVERLOAD_2: -0,
                KPI_OVERLOAD_3: 0,
                KPI_OVERLOAD_4: 0,
                KPI_EXAM_DURA: -200}
    regis_file = "./data/exam_registration.csv"
    timespace_file = "./data/exam_timetable_2023.csv"
    # optimize
    exam_opt = GAOptimizer()
    exam_opt.initialize(timespace_file, rooms, room_caps, fixed_exams, regis_file, "CID")
    exam_opt.optimize(kpi_coef, pop_size=100, mutation_rate=0.4, num_generation=800)
    print(exam_opt.get_feasibility())
    print(exam_opt.get_kpis())
    exam_opt.output_table()

    # # print results to console
    # exam_opt.print_result(exam_table=True, student_statistic=True)
    # # save results to local files
    # exam_opt.output_time_table(student_specific=True)
