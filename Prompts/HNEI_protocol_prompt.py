class HNEI_protocol_prompt():
    Protocols_prompts = {
        1.0:  # HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_a
            f"Battery specifications: The data comes from a lithium-ion battery in a format of 18650 cylindrical cell. "
            f"Its positive electrode is a mixture of LiCoO2 and LiNi4Co4Mn2O2. "
            f"Its negative electrode is graphitic intercalation compound. "
            f"The electrolyte formula is unknown. "
            f"The battery manufacturer is LG Chemical Limited. "
            f"The nominal capacity is 2.8 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 25 degree Celsius. "
            f"The cycling consists of three different strategies. "
            f"For 1st to 10th and 12th to 15th cycles, the battery was charged at a constant current of 0.5 C until reaching 4.3 V, "
            f"then was discharged at a constant current of 0.5 C until reaching 3 V."
            f"For 11th and 16th cycles, the battery was charged at a constant current of 1 C until reaching 4.3 V, "
            f"then was discharged at a constant current of 1 C until reaching 3 V. "
            f"For 17th cycle and the cycles after that, the battery was charged at a constant current of 0.5 C until reaching 4.3 V, "
            f"then was discharged at a constant current of 1.5 C until reaching 3 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n",
        2.0:  # HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_b---HNEI_18650_NMC_LCO_25C_0-100_0.5-1.5C_t
            f"Battery specifications: The data comes from a lithium-ion battery in a format of 18650 cylindrical cell. "
            f"Its positive electrode is a mixture of LiCoO2 and LiNi4Co4Mn2O2. "
            f"Its negative electrode is graphitic intercalation compound. "
            f"The electrolyte formula is unknown. "
            f"The battery manufacturer is LG Chemical Limited. "
            f"The nominal capacity is 2.8 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 25 degree Celsius. "
            f"The cycling consists of three different strategies. "
            f"For 1st to 10th and 12th to 15th cycles, the battery was charged at a constant current of 0.5 C until reaching 4.3 V, "
            f"then was discharged at a constant current of 0.5 C until reaching 3 V."
            f"For 11th cycle, the battery was charged at a constant current of 1 C until reaching 4.3 V, "
            f"then was discharged at a constant current of 1 C until reaching 3 V. "
            f"For 12th cycle and the cycles after that, the battery was charged at a constant current of 0.5 C until reaching 4.3 V, "
            f"then was discharged at a constant current of 1.5 C until reaching 3 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n",
    }