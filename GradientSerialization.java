package learning_hamr;
import java.io.IOException;

import com.etinternational.hamr.io.*;

public class GradientSerialization extends Serialization<Gradient> {

    public GradientSerialization() {
        super(Gradient.class);
    }
    
    @Override
    public void serialize(Gradient gradient, SerializedOutput output) throws IOException {
        output.writeObject(gradient.col_index);
        output.writeObject(gradient.col_value);
    }
    
    @Override
    public Gradient deserialize(SerializedInput input) throws IOException {
        final int col_index = Integer.parseInt(input.readObject().toString());
        final double col_value = Double.parseDouble(input.readObject().toString());
        return new Gradient(col_index, col_value);
    }
    
    @Override
    public GradientSerialization clone() {
        return new GradientSerialization();
    }  
}