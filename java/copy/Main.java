import java.io.ByteArrayOutputStream;
import java.io.ObjectOutputStream;

/**
 * A
 */
class A {

    public int a;    
}

/**
 * main
 */
public class Main {

    public static void main(String[] args) {
        A a = new A();
        a.a = 1;
        A b = new A();
        b = a;
        b.a = 2;
        System.out.println(a.a);
        System.out.println(b.a);
        a.a = 3;
        System.out.println(a.a);
        System.out.println(b.a);
    }
}