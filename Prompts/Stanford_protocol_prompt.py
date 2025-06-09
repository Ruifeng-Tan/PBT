class Stanford_protocol_prompt():
    Protocols_prompts = {
        1.0:  # 100 Reference cell
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0552 A to 4.09 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 72 hours. ",
        2.0:  ## 101 Reference cell
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 c to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0552 A to 4.09 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 72 hours. ",
        3.0:  # 102 Reference cell
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0552 A to 4.09 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 72 hours. ",
        4.0:  # 191
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0264 A to 3.71 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 72 hours. ",
        5.0:  # 192
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0264 A to 3.71 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 72 hours. ",
        6.0:  # 193
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.1512 A to 3.74 V and then charged at 0.036 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hour. ",
        7.0:  # 194
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.1512 A to 3.74 V and then charged at 0.036 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hour. ",
        8.0:  # 195
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.1512 A to 3.74 V and then charged at 0.036 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hour. ",
        9.0:  # 196
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.3936 A to 3.62 V and then charged at 0.0216 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 72 hours. ",
        10.0:  # 198
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.3936 A to 3.62 V and then charged at 0.0216 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 72 hours. ",
        11.0:  # 199
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.2664 A to 3.96 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        12.0:  # 200
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.2664 A to 3.96 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        13.0:  # 201
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.2664 A to 3.96 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        14.0:  # 202
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.68 V and then charged at 0.18 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        15.0:  # 203
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.68 V and then charged at 0.18 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        16.0:  # 204
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.68 V and then charged at 0.18 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        17.0:  # 205
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0456 A to 3.87 V and then charged at 0.72 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        18.0:  # 206
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0456 A to 3.87 V and then charged at 0.72 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        19.0:  # 207
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0456 A to 3.87 V and then charged at 0.72 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        20.0:  # 208
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.18 A to 3.89 V and then charged at 0.0504 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        21.0:  # 209
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.18 A to 3.89 V and then charged at 0.0504 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        22.0:  # 210
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.18 A to 3.89 V and then charged at 0.0504 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        23.0:  # 211
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.73 V and then charged at 0.0504 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        24.0:  # 212
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.73 V and then charged at 0.0504 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        25.0:  # 213
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.73 V and then charged at 0.0504 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        26.0:  # 214
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.2376 A to 4.04 V and then charged at 0.1512 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        27.0:  # 215
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.2376 A to 4.04 V and then charged at 0.1512 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        28.0:  # 216
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.2376 A to 4.04 V and then charged at 0.1512 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        29.0:  # 217
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.2184 A to 3.82 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        30.0:  # 219
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.2184 A to 3.82 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        31.0:  # 220
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 3.86 V and then charged at 0.2544 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        32.0: # 221
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 3.86 V and then charged at 0.2544 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        33.0: # 222
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 3.86 V and then charged at 0.2544 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        34.0: # 223
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.036 A to 3.69 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        35.0: # 224
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.036 A to 3.69 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        36.0: # 225
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.036 A to 3.69 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        37.0:  # 226
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0744 A to 4.08 V and then charged at 0.0936 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        38.0:  # 227
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0744 A to 4.08 V and then charged at 0.0936 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        39.0:  # 228
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0744 A to 4.08 V and then charged at 0.0936 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        40.0:  # 229
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.1128 A to 3.7 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        41.0:  # 269
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.1128 A to 3.7 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        42.0:  # 270
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 4.03 V and then charged at 0.312 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        43.0:  # 271
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 4.03 V and then charged at 0.312 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        44.0:  # 272
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 4.03 V and then charged at 0.312 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        45.0:  # 273
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.95 V and then charged at 0.1368 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        46.0:  # 274
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.95 V and then charged at 0.1368 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        47.0:  # 275
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.95 V and then charged at 0.1368 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        48.0:  # 276
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0936 A to 3.92 V and then charged at 0.2256 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        49.0:  # 277
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0936 A to 3.92 V and then charged at 0.2256 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        50.0:  # 278
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0936 A to 3.92 V and then charged at 0.2256 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        51.0:  # 279
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.94 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        52.0:  # 280
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.94 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        53.0:  # 281
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.94 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        54.0:  # 282
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.132 A to 3.63 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        55.0:  # 283
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.132 A to 3.63 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        56.0:  # 284
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.132 A to 3.63 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        57.0:  # 285
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.78 V and then charged at 0.0216 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 72 hour. ",
        58.0:  # 286
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.78 V and then charged at 0.0216 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 72 hour. ",
        59.0:  # 287
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.78 V and then charged at 0.0216 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 72 hour. ",
        60.0:  # 288
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.18 A to 3.74 V and then rested for 12 hours and then charged at 0.4 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        61.0:  # 289
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.18 A to 3.74 V and then rested for 12 hours and then charged at 0.4 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        62.0:  # 290
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.18 A to 3.74 V and then rested for 12 hours and then charged at 0.4 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        63.0:  # 291
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.53 A to 4.1 V and then rested for 12 hours and then charged at 0.49 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        64.0:  # 292
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.53 A to 4.1 V and then rested for 12 hours and then charged at 0.49 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        65.0:  # 293
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.53 A to 4.1 V and then rested for 12 hours and then charged at 0.49 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        66.0:  # 294
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.4 A to 3.65 V and then rested for 12 hours and then charged at 0.27 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        67.0:  # 295
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.4 A to 3.65 V and then rested for 12 hours and then charged at 0.27 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        68.0:  # 296
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.4 A to 3.65 V and then rested for 12 hours and then charged at 0.27 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        69.0:  # 297
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.22 A to 3.83 V and then rested for 12 hours and then charged at 0.31 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        70.0:  # 299
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.22 A to 3.83 V and then rested for 12 hours and then charged at 0.31 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        71.0:  # 300
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.44 A to 4.05 V and then rested for 12 hours and then charged at 0.53 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        72.0:  # 301
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.44 A to 4.05 V and then rested for 12 hours and then charged at 0.53 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        73.0:  # 302
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.44 A to 4.05 V and then rested for 12 hours and then charged at 0.53 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        74.0:  # 303
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.66 A to 3.92 V and then rested for 12 hours and then charged at 0.57 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        75.0:  # 304
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.66 A to 3.92 V and then rested for 12 hours and then charged at 0.57 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        76.0:  # 305
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.66 A to 3.92 V and then rested for 12 hours and then charged at 0.57 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        77.0:  # 306
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.57 A to 3.78 V and then rested for 12 hours and then charged at 0.66 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        78.0:  # 307
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.57 A to 3.78 V and then rested for 12 hours and then charged at 0.66 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        79.0:  # 308
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.57 A to 3.78 V and then rested for 12 hours and then charged at 0.66 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        80.0:  # 309
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.62 A to 3.6 V and then rested for 12 hours and then charged at 0.44 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        81.0:  # 310
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.62 A to 3.6 V and then rested for 12 hours and then charged at 0.44 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        82.0: # 311
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.62 A to 3.6 V and then rested for 12 hours and then charged at 0.44 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        83.0: # 312
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.49 A to 4.01 V and then rested for 12 hours and then charged at 0.18 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        84.0: # 313
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.49 A to 4.01 V and then rested for 12 hours and then charged at 0.18 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        85.0: # 314
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.49 A to 4.01 V and then rested for 12 hours and then charged at 0.18 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        86.0: # 315
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.31 A to 3.69 V and then rested for 12 hours and then charged at 0.62 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        87.0:  # 316
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.31 A to 3.69 V and then rested for 12 hours and then charged at 0.62 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        88.0:  # 317
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.31 A to 3.69 V and then rested for 12 hours and then charged at 0.62 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        89.0:  # 318
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.27 A to 3.87 V and then rested for 12 hours and then charged at 0.22 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        90.0:  # 319
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.27 A to 3.87 V and then rested for 12 hours and then charged at 0.22 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        91.0:  # 320
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.27 A to 3.87 V and then rested for 12 hours and then charged at 0.22 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        92.0:  # 321
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.35 A to 3.96 V and then rested for 12 hours and then charged at 0.35 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        93.0:  # 322
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.35 A to 3.96 V and then rested for 12 hours and then charged at 0.35 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        94.0:  # 323
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.35 A to 3.96 V and then rested for 12 hours and then charged at 0.35 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 1 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        95.0:  # 324
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.012 A to 3.8 V and then charged at 0.012 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        96.0:  # 325
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.012 A to 3.8 V and then charged at 0.012 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        97.0:  # 326
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.012 A to 3.8 V and then charged at 0.012 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        98.0:  # 103
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 4.06 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        99.0:  # 104
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 4.06 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        100.0:  # 105
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 4.06 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        101.0:  # 106
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.2952 A to 4.1 V and then charged at 0.2832 A to 4.4 V. "
            f"After that, a 1-hour constant voltage hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 168 hours. ",
        102.0:  ## 107
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.2952 A to 4.1 V and then charged at 0.2832 A to 4.4 V. "
            f"After that, a 1-hour constant voltage hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 168 hours. ",
        103.0:  # 108
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.2952 A to 4.1 V and then charged at 0.2832 A to 4.4 V. "
            f"After that, a 1-hour constant voltage hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 168 hours. ",
        104.0:  # 109
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.084 A to 3.97 V and then charged at 0.108 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 168 hours. ",
        105.0:  # 110
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.084 A to 3.97 V and then charged at 0.108 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 168 hours. ",
        106.0:  # 112
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.1224 A to 3.91 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 72 hour. ",
        107.0:  # 113
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.1224 A to 3.91 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 72 hour. ",
        108.0:  # 114
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.1224 A to 3.91 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 72 hour. ",
        109.0:  # 115
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0264 A to 3.72 V and then charged at 0.036 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        110.0:  # 116
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0264 A to 3.72 V and then charged at 0.036 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        111.0:  # 117
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0264 A to 3.72 V and then charged at 0.036 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        112.0:  # 118
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0264 A to 3.72 V and then charged at 0.036 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        113.0:  # 119
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0264 A to 3.72 V and then charged at 0.036 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        114.0:  # 120
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0264 A to 3.72 V and then charged at 0.036 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        115.0:  # 121
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.84 V and then charged at 0.0504 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        116.0:  # 122
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.84 V and then charged at 0.0504 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        117.0:  # 123
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.84 V and then charged at 0.0504 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        118.0:  # 124
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.432 A to 3.66 V and then charged at 0.036 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        119.0:  # 125
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.432 A to 3.66 V and then charged at 0.036 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        120.0:  # 126
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.432 A to 3.66 V and then charged at 0.036 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        121.0:  # 127
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0648 A to 3.76 V and then charged at 0.4296 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        122.0:  # 128
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0648 A to 3.76 V and then charged at 0.4296 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        123.0:  # 129
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0648 A to 3.76 V and then charged at 0.4296 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        124.0:  # 130
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.3264 A to 3.75 V and then charged at 0.1944 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        125.0:  # 131
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.3264 A to 3.75 V and then charged at 0.1944 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        126.0:  # 134
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.1608 A to 4.05 V and then charged at 0.0792 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        127.0:  # 135
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.1608 A to 4.05 V and then charged at 0.0792 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        128.0:  # 136
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0456 A to 3.8 V and then charged at 0.588 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        129.0:  # 137
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0456 A to 3.8 V and then charged at 0.588 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        130.0:  # 138
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0456 A to 3.8 V and then charged at 0.588 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        131.0:  # 139
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 3.6 V and then charged at 0.3408 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        132.0: # 140
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 3.6 V and then charged at 0.3408 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        133.0: # 141
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 3.6 V and then charged at 0.3408 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        134.0: # 142
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.3552 A to 4 V and then charged at 0.384 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        135.0: # 143
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.3552 A to 4 V and then charged at 0.384 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        136.0: # 144
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.3552 A to 4 V and then charged at 0.384 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        137.0:  # 145
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0456 A to 4.08 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        138.0:  # 146
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0456 A to 4.08 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        139.0:  # 147
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0456 A to 4.08 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        140.0:  # 148
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0648 A to 4.07 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        141.0:  # 149
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0648 A to 4.07 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        142.0:  # 150
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0648 A to 4.07 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        143.0:  # 151
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 3.64 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        144.0:  # 152
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 3.64 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        145.0:  # 154
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.65 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        146.0:  # 155
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.65 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        147.0:  # 156
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.65 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        148.0:  # 157
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.48 A to 3.83 V and then charged at 0.1224 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        149.0:  # 158
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.48 A to 3.83 V and then charged at 0.1224 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        150.0:  # 159
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.48 A to 3.83 V and then charged at 0.1224 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        151.0:  # 160
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.9 V and then charged at 0.036 A to 4.4 V. "
            f"After that, a 1-hour constant voltage hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        152.0:  ## 161
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.9 V and then charged at 0.036 A to 4.4 V. "
            f"After that, a 1-hour constant voltage hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        153.0:  # 162
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 35 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.9 V and then charged at 0.036 A to 4.4 V. "
            f"After that, a 1-hour constant voltage hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        154.0:  # 163
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.1032 A to 3.67 V and then charged at 0.4728 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        155.0:  # 165
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.1032 A to 3.67 V and then charged at 0.4728 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        156.0:  # 166
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.81 V and then charged at 0.0792 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hour. ",
        157.0:  # 167
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.81 V and then charged at 0.0792 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hour. ",
        158.0:  # 168
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.81 V and then charged at 0.0792 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hour. ",
        159.0:  # 169
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.88 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        160.0:  # 170
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.88 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        161.0:  # 171
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
            f"Then, it was charged at 0.0048 A to 3.88 V and then charged at 0.0048 A to 4.4 V. "
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. "
            f"This was followed by a 0.048 A discharge to 3 V. "
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. "
            f"Before cycling, the battery rested for 0 hours. ",
        162.0:  # 172
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0264 A to 3.98 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        163.0:  # 173
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0264 A to 3.98 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        164.0:  # 174
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0264 A to 3.98 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        165.0:  # 175
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 3.61 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        166.0:  # 176
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 3.61 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        167.0:  # 177
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0048 A to 3.61 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        168.0:  # 178
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.99 V and then charged at 0.5304 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        169.0:  # 179
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.99 V and then charged at 0.5304 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        170.0:  # 180
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.99 V and then charged at 0.5304 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 168 hours. ", # time of rest
        171.0:  # 181
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.93 V and then charged at 0.0648 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        172.0:  # 182
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.93 V and then charged at 0.0648 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        173.0:  # 183
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0168 A to 3.93 V and then charged at 0.0648 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        174.0:  # 184
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.036 A to 4.02 V and then charged at 0.648 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        175.0:  # 185
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.036 A to 4.02 V and then charged at 0.648 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        176.0:  # 186
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 40 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.036 A to 4.02 V and then charged at 0.648 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 2 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 0 hours. ", # time of rest
        177.0:  # 187
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.1992 A to 3.79 V and then charged at 0.0648 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        178.0:  # 188
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.1992 A to 3.79 V and then charged at 0.0648 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        179.0:  # 189
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.1992 A to 3.79 V and then charged at 0.0648 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        180.0:  # 190
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 45 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.0264 A to 3.71 V and then charged at 0.0048 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 3 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
        181.0:  # 230
            f"Battery specifications: The data comes from a lithium-ion battery in a format of pouch battery. "
            f"Its positive electrode is Li(Ni0.5Co0.2Mn0.3)O2 (NCM523). "
            f"Its negative electrode is artificial graphite. "
            f"The electrolyte formula consists of EC, EMC, DMC, and VC. The composition is 1 M LiPF6 in EC/EMC/DMC (1 : 1 : 1 by volume) solvent with 2% VC (by weight) additive. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 0.24 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 30 degrees Celsius. "
            f"The battery was charged at a constant current of 1 C until reaching 4.4 V followed by a constant voltage hold to 0.05C. "
            f"The battery was then discharged at a constant current of 0.75 C until reaching 3.0 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n"
            f"During formation, at 55 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. " # temperature may be different
            f"Then, it was charged at 0.1128 A to 3.7 V and then charged at 0.0216 A to 4.4 V. " # currents and cutoff voltage may be different
            f"After that, a 1-hour constant voltage (CV) hold was applied at 4.4 V. " 
            f"This was followed by a 0.048 A discharge to 3 V. " 
            f"After the first formation cycle, 5 additional ±0.048 A cycles were performed between 3 and 4.4 V without CV hold or rest periods. " # n verification
            f"Finally, the battery was held at 3 V for 1 hour before being degassed and resealed in an Ar glove box. " 
            f"Before cycling, the battery rested for 72 hours. ", # time of rest
    }
