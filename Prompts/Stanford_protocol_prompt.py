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
            f"During formation, at 25 degrees Celsius, the battery was first charged at 0.2 C to 1.5 V and held for 24 hours. "
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
        41.0:  # 230
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
