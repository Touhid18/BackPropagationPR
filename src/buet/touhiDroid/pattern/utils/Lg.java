package buet.touhiDroid.pattern.utils;

/**
 * Created by touhid on 4/22/15.
 * @author touhid
 */
public class Lg {

    public static void p(String msg){
        System.out.print(msg);
    }

    public static void pl(String msg){
        System.out.println(msg);
    }

    public static void pl(){
        System.out.println();
    }
    public static void e(String msg, Exception e){
        System.out.println(msg);
        e.printStackTrace();
    }
}
