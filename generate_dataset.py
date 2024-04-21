import random
import numpy as np

import matplotlib.pyplot as plt

def generate_raw_dataset_A(num_examples, num_types, max_length, num_instances=1):
    """
    generise za svaki objekat pocetak i duzinu, i generise objekte da budu zalepljeni jedaan do drugog

    dataset = generate_raw_dataset_A(2000, 3, 600)
    Returns
    -------:
    [[start_obj1, length_obj1],[start_obj2, length_obj2],[]]


    """
    max_type_length = int(max_length / num_types)
    data_set = []
    for example_id in range(0, num_examples):
        sum_length = 0
        example = []
        for object_id in range(0, num_types):
            start_position = sum_length if sum_length != 0 else 0
            length = random.randint(10, max_type_length)
            example.append([start_position, length])
            sum_length = sum_length + length
        data_set.append(example)
    return {"objects": np.asarray(data_set)}



# dataset_np = np.asarray(dataset)

def generate_raw_dataset_B(num_examples, num_types, max_length, num_instances=1):
    """
    generise za svaki objekat pocetak i duzinu
    genrise da razmak izmedju svaka dva objekta bude 10, pa 20 pa 30...

    dataset = generate_raw_dataset_B(2000, 3, 600)

    Returns
    -------:
    [[start_obj1, length_obj1],[start_obj2, length_obj2],[]]
    """
    max_length = max_length - (int((num_types * (num_types - 1)) / 2)) * 10
    max_type_length = int(max_length / num_types)
    data_set = []
    for example_id in range(0, num_examples):
        start_position = 0
        example = []
        space = 10
        for object_id in range(0, num_types):
            length = random.randint(10, max_type_length)
            example.append([start_position, length])
            start_position = start_position + length + space
            space = space + 10
        data_set.append(example)
    return {"objects": np.asarray(data_set)}






def generate_raw_dataset_C(num_examples, max_length, num_instances=1):
    """
    generise duzinu prostora i za svaki objekat pocetak i duzinu
    pravi dataset gde je prvi objekat na levom ili desnom kraju i shodno tome se drugi smesta na suprotni kraj
    to sam napravila tako sto stavim neki objekat na pocetak, pa napravim razmak pa smestim neki objekat
    i onda u 50% slucajeva mi je ovaj sto je na pocetku prvi a 50% drugi

    dataset = generate_raw_dataset_C(1000, 600)
    Returns
    -------:
    side, [space_length,[start_obj1, length_obj1],[start_obj2, length_obj2],[]]
    side true znaci da je sa leve strane o2 a side false da je sa desne

    """
    side = []
    num_types = 2
    max_type_length = int(max_length / (num_types + 1))
    data_set = {"space_length": [], "objects": []}
    for example_id in range(0, num_examples):
        left_or_right = random.randint(0, 1)
        space_length = 0

        length = random.randint(10, max_type_length)
        object_left = [space_length, length]
        space_length = space_length + length

        space_between = random.randint(0, max_type_length)
        space_length = space_length + space_between

        length = random.randint(10, max_type_length)
        object_right = [space_length, length]
        space_length = space_length + length

        data_set["space_length"].append(space_length)
        if left_or_right == 0:
            data_set["objects"].append([object_left, object_right])
            side.append(False)
        else:
            data_set["objects"].append([object_right, object_left])
            side.append(True)

    data_set["space_length"] = np.asarray(data_set["space_length"])
    data_set["objects"] = np.asarray(data_set["objects"])

    return side, data_set






def generate_raw_dataset_D(num_examples, max_length, num_instances=1):
    """
    generise duzinu prostora, za svaki objekat pocetak i duzinu
    prvi je negde na sredini a dugi je na levom ili desnom kraju, gde moze da ga smesti
    to sam napravila tako sto stavim drugi objekat na pocetak, pa prvi negde na sredinu i onda izgenerisem ostatak prostora
    da mu duzina bude manja od velicine drugog objekta, i u 50% slucajeva mirrorujem prostor, tako da ovakav
    raspored bude kad gledas s desna na levo

    dataset = generate_raw_dataset_D(1000, 600)
    Returns
    -------:
    side, [space_length,[start_obj1, length_obj1],[start_obj2, length_obj2],[]]
    side true znaci da je sa leve strane o2 a side false da je sa desne

    """
    num_types = 2
    side = []

    max_type_length = int(max_length / (num_types + 2))
    data_set = {"space_length": [], "objects": []}
    for example_id in range(0, num_examples):
        space_length = 0

        length = random.randint(10, max_type_length)
        object2 = [space_length, length]
        space_length = space_length + length

        space_between = random.randint(0, max_type_length)
        space_length = space_length + space_between

        length = random.randint(10, max_type_length)
        object1 = [space_length, length]
        space_length = space_length + length

        space_length = space_length + random.randint(0, object2[1])  # s ove strane nije bilo mesta da se smesti objekat
        data_set["space_length"].append(space_length)

        if random.randint(0, 1) != 0:
            object2_start = space_length - object2[1]
            object2 = [object2_start, object2[1]]
            object1_start = space_length - object1[0] - object1[1]
            object1 = [object1_start, object1[1]]
            side.append(False)
        else:
            side.append(True)
        data_set["objects"].append([object1, object2])

    data_set["space_length"] = np.asarray(data_set["space_length"])
    data_set["objects"] = np.asarray(data_set["objects"])

    return side, data_set





def generate_raw_dataset_E(num_examples, max_length, num_instances=1):
    """
    generise duzinu prostora, za svaki objekat pocetak i duzinu
    prvi je negde na sredini a dugi je na levom ili desnom kraju, gde IMA VISE MEStA
    to sam napravila tako sto stavim drugi objekat na pocetak, pa prvi negde na sredinu i onda izgenerisem ostatak prostora
    da mu duzina bude manja od velicine drugog objekta, i u 50% slucajeva mirrorujem prostor, tako da ovakav
    raspored bude kad gledas s desna na levo

    dataset = generate_raw_dataset_E(1000, 600)
    Returns
    -------:
    side, [space_length,[start_obj1, length_obj1],[start_obj2, length_obj2],[]]
    side true znaci da je sa leve strane o2 a side false da je sa desne
    """
    num_types = 2
    side = []

    max_type_length = int(max_length / (num_types + 2))
    data_set = {"space_length": [], "objects": []}
    while len(data_set["objects"]) < num_examples:
        space_length = 0

        length = random.randint(10, max_type_length)
        object2 = [space_length, length]
        space_length = space_length + length

        space_between = random.randint(0, max_type_length)
        space_length = space_length + space_between
        length = random.randint(10, max_type_length)
        object1 = [space_length, length]
        space_length = space_length + length

        space_length = space_length + random.randint(0, object2[
            1] + space_between - 1)  # s ove strane je bilo manje mesta nego sa druge
        if space_length < max_length:
            data_set["space_length"].append(space_length)

            if random.randint(0, 1) != 0:
                object2_start = space_length - object2[1]
                object2 = [object2_start, object2[1]]
                object1_start = space_length - object1[0] - object1[1]
                object1 = [object1_start, object1[1]]
                side.append(False)
            else:
                side.append(True)
            data_set["objects"].append([object1, object2])

    data_set["space_length"] = np.asarray(data_set["space_length"])
    data_set["objects"] = np.asarray(data_set["objects"])

    return side, data_set





def generate_raw_dataset_F(num_examples, max_length, num_instances=1):
    """
    generise duzinu prostora, za svaki objekat pocetak i duzinu
    prvi je negde na sredini a dugi je zalepljen za njega s leva ako ima mesta ako ne onda je stavljen na desnu ivicu
    to sam napravila tako sto stavim drugi objekat negde na sredinu, pa prvi do njega i onda izgenerisem ostatak prostora
    da mu duzina bude manja od velicine drugog objekta, i u 50% slucajeva mirrorujem prostor, alitako da stavim drugi objekat
    da bude do desne ivice

    dataset = generate_raw_dataset_F(1000, 600)

    Returns
    -------:
    side, [space_length,[start_obj1, length_obj1],[start_obj2, length_obj2],[]]
    side true znaci da je sa leve strane o2 a side fale da je sa desne
    """
    num_types = 2

    side = []
    max_type_length = int(max_length / (num_types + 2))
    data_set = {"space_length": [], "objects": [], "side": []}
    for example_id in range(0, num_examples):
        space_length = 0

        space_between = random.randint(0, max_type_length)
        space_length = space_between
        length = random.randint(10, max_type_length)
        object2 = [space_length, length]
        space_length = space_length + length

        length = random.randint(10, max_type_length)
        object1 = [space_length, length]
        space_length = space_length + length

        space_length = space_length + random.randint(0, object2[1])  # s voe strane nije bilo mesta da se smesti
        data_set["space_length"].append(space_length)

        if random.randint(0, 1) != 0:
            side.append(False)
            object2_start = space_length - object2[1]
            object2 = [object2_start, object2[1]]
            object1_start = space_length - object1[0] - object1[1]
            object1 = [object1_start, object1[1]]
        else:
            side.append(True)
        data_set["objects"].append([object1, object2])

    data_set["space_length"] = np.asarray(data_set["space_length"])
    data_set["objects"] = np.asarray(data_set["objects"])
    data_set["side"] = np.asarray(side)
    return side, data_set




def generate_raw_dataset_G(num_examples, max_length, num_instances=1):
    """
    generise duzinu prostora, za svaki objekat pocetak i duzinu
    prvi je negde na sredini a dugi je zalepljen za njega s leva ili desna, gde ima mesta
    to sam napravila tako sto stavim drugi objekat negde na sredinu, pa prvi do njega i onda izgenerisem ostatak prostora
    da mu duzina bude manja od velicine drugog objekta, i u 50% slucajeva mirrorujem prostor, tako da ovakav
    raspored bude kad gledas s desna na levo

    dataset = generate_raw_dataset_G(1000, 600)

    Returns
    -------:
    [space_length,[start_obj1, length_obj1],[start_obj2, length_obj2],[]]
    """
    num_types = 2
    side = []
    max_type_length = int(max_length / (num_types + 2))
    data_set = {"space_length": [], "objects": []}
    for example_id in range(0, num_examples):
        space_length = 0

        space_between = random.randint(0, max_type_length)
        space_length = space_between
        length = random.randint(10, max_type_length)
        object2 = [space_length, length]
        space_length = space_length + length

        length = random.randint(10, max_type_length)
        object1 = [space_length, length]
        space_length = space_length + length

        space_length = space_length + random.randint(0, object2[1])  # s voe strane nije bilo mesta da se smesti
        data_set["space_length"].append(space_length)

        if random.randint(0, 1) != 0:
            side.append(False)
            object2_start = space_length - object2[1] - object2[0]
            object2 = [object2_start, object2[1]]
            object1_start = space_length - object1[0] - object1[1]
            object1 = [object1_start, object1[1]]
        else:
            side.append(True)
        data_set["objects"].append([object1, object2])

    data_set["space_length"] = np.asarray(data_set["space_length"])
    data_set["objects"] = np.asarray(data_set["objects"])
    data_set["side"] = np.asarray(side)
    return side, data_set




def generate_raw_dataset_H(num_examples, max_length):
    """
    treba da nauci da predvidja poziciju drugog objekta u odnosu na prvi a treceg u odnosu na prvi i drugi
    generise duzinu prostora i  za svaki objekat pocetak i duzinu
    prvi je negde na sredini drugi je na jednom ii drugom kraju sto udaljeniji od prvog, a treci je zalepljen za prvi
    ali sto udaljeniji od drugog
    to sam napravila tako sto stavim drugi objekat na pocetak pa prostor izmedju pa u 50% slucajeva treci objekat pa prvi nalepljen
    na njega i na kraju prostor da bude manji od (treceg objekta) i (drugog + prostor izmedju prvog i drugog),
    a u 50% slucajeva ide drugi pa prvi pa treci a  prostora izmedju prvog i kraja da bude manji od d
    rugog plus prostor izmedju prvog i drugog

    i u 50% slucajeva mirrorujem prostor, tako da ovakav
    raspored bude kad gledas s desna na levo

    dataset = generate_raw_dataset_H(1000, 600)

    Returns
    -------:
    [space_length,[start_obj1, length_obj1],[start_obj2, length_obj2],[]]
    """
    num_types = 3
    max_type_length = int(max_length / (num_types + 2))
    data_set = {"space_length": [], "objects": []}
    while len(data_set["objects"]) < num_examples:
        space_length = 0

        length = random.randint(10, max_type_length)
        object2 = [space_length, length]
        space_length = space_length + length
        space_between = random.randint(0, max_type_length)
        space_length = space_length + space_between

        if random.randint(0, 1) != 0:  # ovde je o3 sa desne strane o1

            length = random.randint(10, max_type_length)
            object1 = [space_length, length]
            space_length = space_length + length

            length = random.randint(10, max_type_length)
            object3 = [space_length, length]
            space_length = space_length + length
            if object3[1] > object2[1] + space_between - object3[1]:
                continue
            space_at_the_end = random.randint(0, min(max_length - space_length,
                                                     object2[1] + space_between - object3[1]))  #
            # ne sme prostor do kraja da bude veci od duzine prostora ili da ispadne da je prostor izmedju prvog i kraja duzi nego
            # izmedju prvog i pocetka(jer bi u tom slucaju drugi objekat bio stavljen na desnu ivicu)
            space_length = space_length + space_at_the_end
        else:  # ovde je o3 sa leve stane o1

            length = random.randint(10, max_type_length)
            object3 = [space_length, length]
            space_length = space_length + length

            length = random.randint(10, max_type_length)
            object1 = [space_length, length]
            space_length = space_length + length

            space_at_the_end = random.randint(0, min(object3[1], object2[1] + space_between + object3[1]))
            space_length = space_length + space_at_the_end  # s ove strane nije bilo mesta da se smesti

        if random.randint(0, 1) != 0:
            object2_start = space_length - object2[1] - object2[0]
            object2 = [object2_start, object2[1]]
            object1_start = space_length - object1[0] - object1[1]
            object1 = [object1_start, object1[1]]
            object3_start = space_length - object3[0] - object3[1]
            object3 = [object3_start, object3[1]]

        data_set["objects"].append([object1, object2, object3])
        data_set["space_length"].append(space_length)

    data_set["space_length"] = np.asarray(data_set["space_length"])
    data_set["objects"] = np.asarray(data_set["objects"])

    return data_set



def generate_raw_dataset_H(num_examples, max_length):
    """
    treba da nauci da predvidja poziciju drugog objekta u odnosu na prvi a treceg u odnosu na prvi i drugi
    generise duzinu prostora i  za svaki objekat pocetak i duzinu
    prvi je negde na sredini drugi je na jednom ii drugom kraju sto udaljeniji od prvog, a treci je zalepljen za prvi
    ali sto udaljeniji od drugog
    to sam napravila tako sto stavim drugi objekat na pocetak pa prostor izmedju pa u 50% slucajeva treci objekat pa prvi nalepljen
    na njega i na kraju prostor da bude manji od (treceg objekta) i (drugog + prostor izmedju prvog i drugog),
    a u 50% slucajeva ide drugi pa prvi pa treci a  prostora izmedju prvog i kraja da bude manji od d
    rugog plus prostor izmedju prvog i drugog

    i u 50% slucajeva mirrorujem prostor, tako da ovakav
    raspored bude kad gledas s desna na levo

    dataset = generate_raw_dataset_H(1000, 600)

    Returns
    -------:
    [space_length,[start_obj1, length_obj1],[start_obj2, length_obj2],[]]
    """
    num_types = 3
    max_type_length = int(max_length / (num_types + 2))
    data_set = {"space_length": [], "objects": []}
    while len(data_set["objects"]) < num_examples:
        space_length = 0

        length = random.randint(10, max_type_length)
        object2 = [space_length, length]
        space_length = space_length + length
        space_between = random.randint(0, max_type_length)
        space_length = space_length + space_between

        if random.randint(0, 1) != 0:  # ovde je o3 sa desne strane o1

            length = random.randint(10, max_type_length)
            object1 = [space_length, length]
            space_length = space_length + length

            length = random.randint(10, max_type_length)
            object3 = [space_length, length]
            space_length = space_length + length
            if object3[1] > object2[1] + space_between - object3[1]:
                continue
            space_at_the_end = random.randint(0, min(max_length - space_length,
                                                     object2[1] + space_between - object3[1]))  #
            # ne sme prostor do kraja da bude veci od duzine prostora ili da ispadne da je prostor izmedju prvog i kraja duzi nego
            # izmedju prvog i pocetka(jer bi u tom slucaju drugi objekat bio stavljen na desnu ivicu)
            space_length = space_length + space_at_the_end
        else:  # ovde je o3 sa leve stane o1

            length = random.randint(10, max_type_length)
            object3 = [space_length, length]
            space_length = space_length + length

            length = random.randint(10, max_type_length)
            object1 = [space_length, length]
            space_length = space_length + length

            space_at_the_end = random.randint(0, min(object3[1], object2[1] + space_between + object3[1]))
            space_length = space_length + space_at_the_end  # s ove strane nije bilo mesta da se smesti

        if random.randint(0, 1) != 0:
            object2_start = space_length - object2[1] - object2[0]
            object2 = [object2_start, object2[1]]
            object1_start = space_length - object1[0] - object1[1]
            object1 = [object1_start, object1[1]]
            object3_start = space_length - object3[0] - object3[1]
            object3 = [object3_start, object3[1]]

        data_set["objects"].append([object1, object2, object3])
        data_set["space_length"].append(space_length)

    data_set["space_length"] = np.asarray(data_set["space_length"])
    data_set["objects"] = np.asarray(data_set["objects"])

    return data_set


def generate_raw_dataset_I(num_examples, max_length):
    """
    treba da nauci da preedvidja poziciju treceg
    generise duzinu prostora, za svaki objekat pocetak i duzinu
    treci treba da bude sto udaljeniji od prvog i drugog, ako ima mesta smesti ga na ivicu ako nema onda izmedju
    posmatrala sam dva slucaja, jedan je kad treci treba da s smesti izmedju prvog i drugog a drugi
    je kad je treci na nekom kraju

    at_the_edge, dataset = generate_raw_dataset_I(1000, 600)

    Returns
    -------:
    at_the edge, [space_length,[start_obj1, length_obj1],[start_obj2, length_obj2],[]]
    at_the true znaci da 03 na nekoj ivici
    """
    num_types = 3
    at_the_edge = []

    max_type_length = int(max_length / (num_types + 2))
    data_set = {"space_length": [], "objects": []}

    while len(data_set["objects"]) < num_examples:
        space_length = 0
        if random.randint(0, 1) != 0:  # treci je na pocetku pa onda idu prvi i drugi

            at_the_edge_one = True
            length = random.randint(10, max_type_length)
            object3 = [space_length, length]
            space_length = space_length + length

            space_between = random.randint(0, max_type_length)
            s1 = space_between + object3[
                1]  ## s1 mi sluzi da kazem koliko sme parce izmedju desne ivice i poslednjeg objekta da bude,
            # treba u ovom slucaju da bude manje od duzine o3 i prostora izmedju o3 i o1, jer bi onda mogao da ga smesti
            # na drugoj ivici
            space_length = space_length + space_between
            length = random.randint(10, max_type_length)
            object1 = [space_length, length]
            space_length = space_length + length

            space_between = random.randint(0, max_type_length)
            space_length = space_length + space_between
            length = random.randint(10, max_type_length)
            object2 = [space_length, length]
            space_length = space_length + length

        else:  # treci je izmedju

            at_the_edge_one = False

            space_length = random.randint(0, max_type_length)
            s1 = space_length
            length = random.randint(10, max_type_length)
            object1 = [space_length, length]
            space_length = space_length + length

            space_between = random.randint(0, max_type_length)
            space_length = space_length + space_between
            length = random.randint(max(s1, 10), max_type_length)  ### treba 03 da bude veci od prostora do ivice,
            # jer bi u suprotonm bio tamo smesten

            object3 = [space_length, length]
            space_length = space_length + length

            space_length = space_length + space_between
            length = random.randint(10, max_type_length)
            object2 = [space_length, length]
            space_length = space_length + length

        if random.randint(0, 1) != 0:
            temp = object1
            object1 = object2
            object2 = temp

        if space_length < max_length:
            at_the_edge.append(at_the_edge_one)
            space_length = space_length + random.randint(0, min(s1, max_length - space_length))

            if random.randint(0, 1) != 0:
                object2_start = space_length - object2[1] - object2[0]
                object2 = [object2_start, object2[1]]
                object1_start = space_length - object1[0] - object1[1]
                object1 = [object1_start, object1[1]]

                object3_start = space_length - object3[0] - object3[1]
                object3 = [object3_start, object3[1]]

            data_set["objects"].append([object1, object2, object3])
            data_set["space_length"].append(space_length)

    data_set["space_length"] = np.asarray(data_set["space_length"])
    data_set["objects"] = np.asarray(data_set["objects"])

    return at_the_edge, data_set




def generate_raw_dataset_J(num_examples, max_length):
    """stavlja o2 da bude sto dalje od o1 a da feature bude iza njega, Predvidja poziciju o2 na osnovu f1 i o1
    postavim o1 i o2, ako je o2 na ivici onda je f1 bilo gde iza o2, ako ne onda je f1, na ivici o2 koja je bliza o1


    dataset = generate_raw_dataset_J(1000, 600)

    """
    num_types = 2

    max_type_length = int(max_length / (num_types + 2))
    data_set = {"space_length": [], "objects": [], "features": [],"types":[]}
    while len(data_set["objects"]) < num_examples:

        space_length = random.randint(0, max_type_length)
        s1 = space_length
        length = random.randint(10, max_type_length)
        object1 = [space_length, length]
        space_length = space_length + length

        space_between = random.randint(0, max_type_length)
        space_length = space_length + space_between
        length = random.randint(max(s1,10), max_type_length)
        object2 = [space_length, length]
        space_length = space_length + length

        if random.randint(0, 1) != 0: # o2 nije na ivici
            typee = 0
            f1_pos = object2[0]
            space_length = space_length + random.randint(0, max_length - space_length)

        else:
            f1_pos = random.randint(object2[0], space_length-1)
            typee = 1

        if random.randint(0, 1) != 0:
            object2_start = space_length - object2[1] - object2[0]
            object2 = [object2_start, object2[1]]
            object1_start = space_length - object1[0] - object1[1]
            object1 = [object1_start, object1[1]]


            f1_pos = space_length - f1_pos - 1

        data_set["objects"].append([object1, object2])
        data_set["space_length"].append(space_length)
        data_set["features"].append(f1_pos)
        data_set["types"].append(typee)

    data_set["space_length"] = np.asarray(data_set["space_length"])
    data_set["objects"] = np.asarray(data_set["objects"])
    data_set["features"] = np.asarray(data_set["features"])
    data_set["types"] = np.asarray(data_set["types"])

    return data_set




def generate_raw_dataset_K(num_examples, max_length, max_allowed_closenest=1):
    """ Predvidja poziciju o2 na osnovu f1 i o1
    traba iti sto blizi o1 ali ne sme f1 da ude iza

    types, dataset = generate_raw_dataset_K(1000, 600)
    """

    num_types = 2
    types = []
    max_type_length = int(max_length / (num_types + 2))
    data_set = {"space_length": [], "objects": [], "features": [],"types":[]}
    while len(data_set["objects"]) < num_examples: ## ovde ih lepi

        layout_id = random.randint(0, 2)
        if layout_id == 0: ### zalepljeni su o1 i o2
            typee = 0
            space_length = random.randint(0, max_type_length)
            length = random.randint(10, max_type_length)
            object2 = [space_length, length]
            space_length = space_length + length

            length = random.randint(10, max_type_length)
            object1 = [space_length, length]
            space_length = space_length + length

            space_between = random.randint(0, max_type_length)
            space_length = space_length + space_between
            f1_pos = space_length

        if layout_id == 1:

            space_length = random.randint(0, max_type_length)
            s1 = space_length
            length = random.randint(10, max_type_length)
            object1 = [space_length, length]
            space_length = space_length + length

            length = random.randint(max(s1,10), max_type_length)
            object2 = [space_length, length]
            space_length = space_length + length

            # ovde simuiram situacije gde prekrsi uslov da je blize f1 od 5 po cenu toga da se zalepi za o1
            # if random.randint(0, 1) == 0:
            #     typee = 1
            #     space_between = random.randint(max_allowed_closenest, max_type_length)
            # else:
            #     typee = 2
            #     space_between = random.randint(0, max_allowed_closenest)

            space_between = random.randint(max_allowed_closenest, max_type_length)

            space_length = space_length + space_between
            f1_pos = space_length
            typee = 1

        if layout_id == 2:         # ne moze da ga zalepi zato sto je s jedne strane mali prostor do iveice, a s druge mali izmedju f1 i o1, i onda mora da ga stavi sa suprotne strane f1 u odnosu na o1
            # e tu bi sustinski trebalo da ga sto vise priblizi f1 i ovo max_allowed_closenest znaci najvise mozes da ga pribizis max_allowed_closenest
            typee = 2
            space_length = random.randint(0, max_type_length)
            s1 = space_length
            length = random.randint(10, max_type_length)
            object1 = [space_length, length]
            space_length = space_length + length

            s2 = random.randint(0, max_type_length)
            space_length = space_length + s2
            f1_pos = space_length

            space_length = space_length + max_allowed_closenest
            length = random.randint(max(s1, s2), max_type_length)
            object2 = [space_length, length]
            space_length = space_length + length

        if space_length < max_length:
            space_length = space_length + random.randint(0, max_length - space_length)

        if random.randint(0, 1) == 0:
            object2_start = space_length - object2[1] - object2[0]
            object2 = [object2_start, object2[1]]
            object1_start = space_length - object1[0] - object1[1]
            object1 = [object1_start, object1[1]]

            f1_pos = space_length - f1_pos - 1

        if space_length < max_length:
            data_set["objects"].append([object1, object2])
            data_set["space_length"].append(space_length)
            data_set["features"].append(f1_pos)
            data_set["types"].append(typee)

    data_set["space_length"] = np.asarray(data_set["space_length"])
    data_set["objects"] = np.asarray(data_set["objects"])
    data_set["features"] = np.asarray(data_set["features"])
    data_set["types"] = np.asarray(data_set["types"])

    return  data_set


