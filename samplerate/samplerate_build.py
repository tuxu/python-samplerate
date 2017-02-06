from cffi import FFI

ffibuilder = FFI()
ffibuilder.set_source('samplerate._src', None)

ffibuilder.cdef("""
    typedef struct SRC_STATE_tag SRC_STATE ;

    typedef struct
    {	const float	*data_in ;
        float	 *data_out ;

        long	input_frames, output_frames ;
        long	input_frames_used, output_frames_gen ;

        int		end_of_input ;

        double	src_ratio ;
    } SRC_DATA ;

    // Simple API
    int src_simple (SRC_DATA *data, int converter_type, int channels) ;

    // Full API
    SRC_STATE* src_new (int converter_type, int channels, int *error) ;
    SRC_STATE* src_delete (SRC_STATE *state) ;
    int src_process (SRC_STATE *state, SRC_DATA *data) ;
    int src_reset (SRC_STATE *state) ;
    int src_set_ratio (SRC_STATE *state, double new_ratio) ;
    int src_is_valid_ratio (double ratio) ;

    // Callback API
    typedef long (*src_callback_t) (void *cb_data, float **data) ;
    SRC_STATE* src_callback_new (src_callback_t func,
                            int converter_type, int channels,
                            int *error, void* cb_data) ;
    long src_callback_read (SRC_STATE *state, double src_ratio,
                            long frames, float *data) ;

    // Extern "Python"-style callback dropped in favor of compiler-less
    // ABI mode ...
    //extern "Python" long src_input_callback(void*, float**);

    // Misc
    int src_error (SRC_STATE *state) ;
    const char* src_strerror (int error) ;
    const char *src_get_name (int converter_type) ;
    const char *src_get_description (int converter_type) ;
    const char *src_get_version (void) ;
    void src_short_to_float_array (const short *in, float *out, int len) ;
    void src_float_to_short_array (const float *in, short *out, int len) ;
    void src_int_to_float_array (const int *in, float *out, int len) ;
    void src_float_to_int_array (const float *in, int *out, int len) ;
""")

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
