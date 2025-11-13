class CALCE_protocol_prompt():
    Protocols_prompts = {
        1.0:  # CALCE_CS2_33, CALCE_CS2_34
            f"Battery specifications: The data comes from a lithium-ion battery in a format of prismatic battery. "
            f"Its positive electrode is a LiCoO2 (LCO). "
            f"Its negative electrode is graphite. "
            f"The electrolyte formula is unknown. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 1.1 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 25 degrees Celsius. "
            f"In the cycling, the battery was charged at a constant current of 0.5 C until reaching 4.2 V. "
            f"The battery was then discharged at a constant current of 0.5 C until reaching 2.7 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n",
        2.0:  # CALCE_CS2_35---38
            f"Battery specifications: The data comes from a lithium-ion battery in a format of prismatic battery. "
            f"Its positive electrode is a LiCoO2 (LCO). "
            f"Its negative electrode is graphite. "
            f"The electrolyte formula is unknown. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 1.1 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 25 degrees Celsius. "
            f"In the cycling, the battery was charged at a constant current of 0.5 C until reaching 4.2 V. "
            f"The battery was then discharged at a constant current of 1 C until reaching 2.7 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n",
        3.0:  # CALCE_CX2_16, CX2_33, CX2_35
            f"Battery specifications: The data comes from a lithium-ion battery in a format of prismatic battery. "
            f"Its positive electrode is a LiCoO2 (LCO). "
            f"Its negative electrode is graphite. "
            f"The electrolyte formula is unknown. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 1.35 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 25 degrees Celsius. "
            f"In the cycling, the battery was charged at a constant current of 0.5 C until reaching 4.2 V. "
            f"The battery was then discharged at a constant current of 0.5 C until reaching 2.7 V. "
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n",
        4.0:  # CALCE_CX2_34, CX2_36, CX2_37, CX2_38
            f"Battery specifications: The data comes from a lithium-ion battery in a format of prismatic battery. "
            f"Its positive electrode is a LiCoO2 (LCO). "
            f"Its negative electrode is graphite. "
            f"The electrolyte formula is unknown. "
            f"The battery manufacturer is unknown. "
            f"The nominal capacity is 1.35 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 25 degrees Celsius. "
            f"In the cycling, the battery was charged at a constant current of 0.5 C until reaching 4.2 V. "
            f"The battery was then discharged at a constant current of 1 C until reaching 2.7 V. " # the discharged current rate should be 0.5C, but we keep 1C here to ensure reproducibility with previous results.
            f"The cycling state-of-charge of this battery ranges from 0% to 100%.\n",
    }