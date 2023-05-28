class Debug:
    def __init__(self, debug_println_number = 20, is_debugging = True):
        self.debug_println_count = 0
        self.debug_println_number = debug_println_number
        self.is_debugging = is_debugging
    
    def print_start(self, I):
        if self.is_debugging:
            print('\n')
            print('START BILATERAL FILTERING ON IMG, SHAPE ({}, {}, {}) **********'.format(I.shape[0], I.shape[1], I.shape[2]))
    
    def print_G_S(self, G_s):
        if self.is_debugging:
            print('Constant G_s for this Gaussian kernel is:')
            print(G_s)
    
    def print_progress(self, I):
        h, w, _ = I.shape
        self.debug_println_count += 1
        interval =  (h * w) // self.debug_println_number
        if self.debug_println_count % interval == 0:
            print('in progress...  {} trail  {}% complete'.format(self.debug_println_count, self.debug_println_count * 100 // (h * w) ))

    def print_pick_pixels_progress(self):
        print('Preprocessing...     picking star pixels')

    def print_expand_border_progress(self):
        print('Preprocessing... expanding star borders')

