import csv
import random
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd
from prettytable import PrettyTable

from deap import algorithms
from deap import base
from deap import creator
from deap import tools


def evaluate_timetable_for_students(arranged_exams: list, fixed_exams: dict, bindings: dict, students: dict,
                                    available_dates: list, week_date_dict: dict):
    # combine three types of exams to get complete exam timetable
    exam_date_table = {exam: date for exam, date in zip(arranged_exams, available_dates) if exam != "no_exam_today"}
    for exam, date in fixed_exams.items():
        exam_date_table[exam] = date
    for exam_k, exam_v in bindings.items():
        exam_date_table[exam_k] = exam_date_table[exam_v]

    # calculate indicators - 2/3/4/5 consecutive exams and 4 exams a week
    consecutive_exams = {2: 0, 3: 0, 4: 0, 5: 0}
    having_4_exams_a_week = 0

    for _, registered_exams in students.items():
        student_exam_dates = [exam_date_table[exam] for exam in registered_exams]
        student_exam_dates.sort()
        # check whether this student has 2/3/4/5 consecutive exams
        consecutive_count = 1
        for pre_exam, next_exam in zip(student_exam_dates[:-1], student_exam_dates[1:]):
            if next_exam - pre_exam <= 1:
                consecutive_count += 1
            else:
                if consecutive_count != 1:
                    consecutive_exams[consecutive_count] += 1
                    consecutive_count = 1
        if consecutive_count > 1:
            consecutive_exams[consecutive_count] += 1
        # check whether this student has 4 exams in a week
        student_exam_dates_set = set(student_exam_dates)
        for week in week_date_dict:
            if len(student_exam_dates_set.intersection(set(week_date_dict[week]))) == 4:
                having_4_exams_a_week += 1

    return consecutive_exams[2], consecutive_exams[3], consecutive_exams[4], consecutive_exams[5], having_4_exams_a_week


def ga_evaluate(individual, fixed_exams: dict, bindings: dict, students: dict,
                available_dates: list, week_date_dict: dict, kpi_weights):
    c_2, c_3, c_4, c_5, e_4 = evaluate_timetable_for_students(individual, fixed_exams, bindings, students,
                                                              available_dates, week_date_dict)
    fitness = kpi_weights["2 consecutive exams"] * c_2 + kpi_weights["3 consecutive exams"] * c_3 + kpi_weights[
        "4 consecutive exams"] * c_4 + kpi_weights["5 consecutive exams"] * c_5 + kpi_weights["4 exams a week"] * e_4
    return fitness,


class Optimizer:
    def __init__(self):
        self.available_dates = None
        self.week_date_dict = None
        # processed data
        self.exams = None  # {exam_code: students}
        self.students = None # {student_id: selected_exams}
        self.no_overlap_exams_pairs = None
        # all exams are classified into three categories - bound, fixed and arranged
        self.bindings = dict()  # the "key" exam is bound with the "value" exam
        self.fixed_exams = dict()  # key is exam, value is date
        self.arranged_exams = list()  # a sequence of exams need to be (or have been) arranged
        # ga_results
        self.ga_pop = None
        self.ga_log = None
        self.exam_date_table = None  # optimised and complete exam timetable

    def initialize(self, available_dates, week_date_dict, fixed_exams, regis_datafile=None, id_column=None):
        self.available_dates = available_dates
        self.week_date_dict = week_date_dict
        self.fixed_exams = fixed_exams

        if regis_datafile is not None:
            self.process_register_data(regis_datafile, id_column)

        if self.exams is not None:
            self.check_overlap()
            self.generate_bindings()

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

        self.students = students
        self.exams = exams
        return students, exams

    def check_overlap(self):
        no_overlap_exams_pairs = []
        for i, j in combinations(self.exams.keys(), 2):
            if not set.intersection(self.exams[i], self.exams[j]):
                no_overlap_exams_pairs.append([i, j])

        self.no_overlap_exams_pairs = no_overlap_exams_pairs
        return no_overlap_exams_pairs

    def generate_bindings(self):
        """ Compress the exams - put 2 non-conflicting exams on the same day"""
        bindings = {}  # the "key" exam will follow the "value" exam
        free_exams = set(self.exams.keys())  # exams have not been bound with others
        for exam_1, exam_2 in self.no_overlap_exams_pairs:
            if exam_1 in free_exams and exam_2 in free_exams:
                if exam_1 in self.fixed_exams:  # the "key" exam cannot be a fixed exam
                    bindings[exam_2] = exam_1
                else:
                    bindings[exam_1] = exam_2
                free_exams.remove(exam_1)
                free_exams.remove(exam_2)

        self.bindings = bindings
        return bindings

    def optimize(self, kpi_weights, pop_size=100, crossover_rate=0, mutation_rate=0.5, num_generation=200):

        free_exams = list(set(self.exams.keys()) - set(self.fixed_exams) - set(self.bindings.keys()))
        if len(free_exams) > len(self.available_dates):
            raise ValueError("the number of exams exceeds the number of available dates")
        else:
            num_no_exam = len(self.available_dates) - len(free_exams)
            free_exams += ["no_exam_today"] * num_no_exam

        # define chromosome and individual
        creator.create("Fitness", base.Fitness, weights=(1,))
        creator.create("Individual", list, fitness=creator.Fitness)
        toolbox = base.Toolbox()
        toolbox.register("chromosome", random.sample, free_exams, len(free_exams))
        toolbox.register("individual", tools.initIterate, creator.Individual,
                         toolbox.chromosome)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        # define evaluation, mutation, crossover and selection methods
        toolbox.register("evaluate", ga_evaluate, fixed_exams=self.fixed_exams, bindings=self.bindings,
                         students=self.students,
                         available_dates=self.available_dates, week_date_dict=self.week_date_dict,
                         kpi_weights=kpi_weights)
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

        exam_date_table = self.gen_time_table()
        return exam_date_table

    def gen_time_table(self):
        # combine three types of exams to get complete exam timetable
        exam_date_table = {exam: date for exam, date in zip(self.arranged_exams, self.available_dates) if
                           exam != "no_exam_today"}
        for exam, date in self.fixed_exams.items():
            exam_date_table[exam] = date
        for exam_k, exam_v in self.bindings.items():
            exam_date_table[exam_k] = exam_date_table[exam_v]
        self.exam_date_table = exam_date_table
        return exam_date_table

    def output_time_table(self, overall=True, student_specific=False):

        if overall:
            with open("data/Exam_timetable.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Course", "Exam Date"])
                writer.writerows([[exam, date] for exam, date in self.exam_date_table.items()])

        if student_specific:
            with open("data/Student-specific_exam_timetable.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Student ID", "Exams and Dates",
                                 "Having 2 Consecutive Exams", "Having 3 Consecutive Exams",
                                 "Having 4 Consecutive Exams", "Having 5 Consecutive Exams", "Having 4 Exams a week"])

                for student, registered_exams in self.students.items():
                    kpis = evaluate_timetable_for_students(self.arranged_exams, self.fixed_exams,
                                                           self.bindings, {student: registered_exams},
                                                           self.available_dates, self.week_date_dict)
                    registered_exams_dates ={exam:self.exam_date_table[exam] for exam in registered_exams}

                    writer.writerow([student, registered_exams_dates] + list(kpis))

    def print_result(self, exam_table=True, student_statistic=True, ga_convergence=False):
        import operator

        if exam_table:
            table = PrettyTable()
            table.field_names = ["Course Code", "Exam Date"]
            table.add_rows([[exam, date] for exam, date in self.exam_date_table.items()])
            print(table.get_string(sort_key=operator.itemgetter(0, 1), sortby="Exam Date"))

            # print(table)

        if student_statistic:
            table = PrettyTable()
            total_student_nums = {"3 consecutive exams": 0, "4 (consecutive/not consecutive) exams": 0,
                                  "5 consecutive exams": 0}
            table.field_names = ["Student ID", "Having 2 Consecutive Exams", "Having 3 Consecutive Exams",
                                 "Having 4 Consecutive Exams", "Having 5 Consecutive Exams", "Having 4 Exams a week"]
            for student, registered_exams in self.students.items():
                kpis = evaluate_timetable_for_students(self.arranged_exams, self.fixed_exams,
                                                       self.bindings, {student: registered_exams},
                                                       self.available_dates, self.week_date_dict)
                table.add_row([student] + list(kpis))

                if kpis[2] > 0 or kpis[4] > 0:
                    total_student_nums["4 (consecutive/not consecutive) exams"] += 1
                if kpis[3] > 0:
                    total_student_nums["5 consecutive exams"] += 1
                if kpis[1] > 0:
                    total_student_nums["3 consecutive exams"] += 1

            # print(table)
            print("There are/is {0} student(s) having 3 consecutive exams a week".format(
                total_student_nums["3 consecutive exams"]))
            print("There are/is {0} student(s) having 4 exams a week".format(
                total_student_nums["4 (consecutive/not consecutive) exams"]))
            print("There are/is {0} student(s) having 5 exams a week".format(total_student_nums["5 consecutive exams"]))

        if ga_convergence:
            fitness_records = [record[0] for record in self.ga_log.select("best")]
            plt.xlabel("Number of Generation")
            plt.ylabel("Best fitness")
            plt.plot(fitness_records)
            plt.show()


def main():
    # input
    regis_file = "./student_registration.csv"
    available_dates = (5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24)
    week_date_dict = {1: {5, 6},
                      2: {9, 10, 11, 12, 13},
                      3: {16, 17, 18, 19, 20},
                      4: {23, 24}}
    fixed_exams = {"CIVE97122": 3}
    kpi_weights = {"2 consecutive exams": -1,
                   "3 consecutive exams": -5,
                   "4 consecutive exams": -70,
                   "5 consecutive exams": -100,
                   "4 exams a week": -70}
    # optimize
    exam_opt = Optimizer()
    exam_opt.initialize(available_dates, week_date_dict, fixed_exams, regis_file, "ID")
    exam_opt.optimize(kpi_weights, pop_size=100, mutation_rate=0.3, num_generation=500)

    # print results to console
    exam_opt.print_result(exam_table=True, student_statistic=True)
    # save results to local files
    exam_opt.output_time_table(student_specific=True)


if __name__ == "__main__":
    main()
