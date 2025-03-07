class RWTH_protocol_prompt():
    Protocols_prompts = {
        1.0:  # All RWTH cells' working conditions are the same
            f"Battery specifications: The data comes from a lithium-ion battery in a format of 18650 cylindrical cell. "
            f"Its positive electrode is NMC. "
            f"Its negative electrode is carbon. "
            f"The electrolyte formula is unknown. "
            f"The battery manufacturer is Sanyo/Panasonic. "
            f"The nominal capacity is 2.05 Ah. "
            f"Operating condition: The working history of this battery is just after formation. "
            f"The working ambient temperature of this battery is 25 degree Celsius. "
            f"In the cycling, the battery was discharged at a constant current of 2 C until reaching 3.5 V. "
            f"The battery was then charged at a constant current of 2 C until reaching 3.9 V. "
            f"The cycling state-of-charge of this battery ranges from 20% to 80%.\n"
    }