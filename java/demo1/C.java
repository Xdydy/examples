import java.lang.reflect.Method;
import java.util.ArrayList;



public class C<Node,Fact> {
    public Fact testFact(ArrayList<Node> list) {
        try {
            B b = new B();
            Method getDataMethod = B.class.getMethod("getData", ArrayList.class);
            return (Fact)getDataMethod.invoke(b, list);
        } catch(Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
