from effects.crt_effect import CRTEffect
from effects.vhs_effect import VHSEffect
from effects.analog_circuit_effect import AnalogCircuitEffect

class EffectFactory:
    """Factory for creating effect processors"""
    
    @staticmethod
    def create_effect(effect_name):
        """Create an effect processor based on name"""
        if effect_name == "VHS Glitch":
            return VHSEffect()
        elif effect_name == "CRT TV":
            return CRTEffect()
        elif effect_name == "Analog Circuit":
            return AnalogCircuitEffect()
        else:
            # Default to CRT effect
            return CRTEffect() 