import numpy as np
# https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwik0Lqhhf-BAxW50QIHHeEGA7AQFnoECAwQAQ&url=https%3A%2F%2Fgrouper.ieee.org%2Fgroups%2F802%2F11%2FWorkshops%2F2019-July-Coex%2FCoexistence-Workshop-802-11ax-Overview.pptx&usg=AOvVaw3vVedQjNfa-YpvDpskpJU2&opi=89978449


#Used for single user !!!
def SIG_A1_A2(Frame_format:int , Beam_change:int , UL_DL:int , MCS : np.ndarray, DMC: np.ndarray, BSS_color:np.ndarray, \
          reserved:int , spatial_reuse:np.ndarray, bandwidth:np.ndarray, GI_LTF_size:np.ndarray, Nsts:np.ndarray, \
          TXOP_duration:np.ndarray, coding:int, LDPC_extra_symbol:int, STBC:int, TxBF:int, Pre_FEC_padding:np.ndarray, \
          BE_Disambiguaty:int , reserved2: int , Doppler:int , CRC:np.ndarray, Tail:np.ndarray) -> np.ndarray:
    
    #make a concatenation of all the parameters
    SIG_A1_A2_sequence = np.concatenate([Frame_format,Beam_change,UL_DL,MCS,DMC,BSS_color,reserved,spatial_reuse,bandwidth,GI_LTF_size,Nsts,TXOP_duration,coding,LDPC_extra_symbol,STBC,TxBF,Pre_FEC_padding,BE_Disambiguaty,reserved2,Doppler,CRC,Tail])
    return SIG_A1_A2_sequence