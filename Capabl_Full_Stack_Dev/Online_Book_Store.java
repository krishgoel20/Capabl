import java.util.*;
import java.util.concurrent.locks.*;

class User
{
    private String UserID;
    private String Username;
    private String Password;

    public User(String UserID,String Username,String Password)
    {
        this.UserID = UserID;
        this.Username = Username;
        this.Password = Password;
    }

    public String getUserID()
    {
        return UserID;
    }

    public String getUsername()
    {
        return Username;
    }

    public String getPassword()
    {
        return Password;
    }
}

class Book
{
    private String Title;
    private String Author;
    private double Price;
    private int Stock;

    public Book(String Title,String Author,double Price,int Stock)
    {
        this.Title = Title;
        this.Author = Author;
        this.Price = Price;
        this.Stock = Stock;
    }

    public String getTitle()
    {
        return Title;
    }

    public String getAuthor()
    {
        return Author;
    }

    public double getPrice()
    {
        return Price;
    }

    public int getStock()
    {
        return Stock;
    }

    public void setStock(int stock)
    {
        this.Stock = Stock;
    }
}

class ShoppingCart
{
    private List <Book> Books;

    public ShoppingCart()
    {
        this.Books = new ArrayList <> ();
    }

    public synchronized void addBook(Book book)
    {
        Books.add(book);
    }

    public synchronized void removeBook(Book book)
    {
        Books.remove(book);
    }

    public List <Book> getBooks()
    {
        return Books;
    }
}

class Inventory
{
    private Map <Book,Integer> bookStocks;
    private ReadWriteLock lock;
    private Lock readLock;
    private Lock writeLock;

    public Inventory()
    {
        this.bookStocks = new HashMap <> ();
        this.lock = new ReentrantReadWriteLock();
        this.readLock = lock.readLock();
        this.writeLock = lock.writeLock();
    }

    public void updateStock(Book book,int newStock)
    {
        writeLock.lock();
        try
        {
            bookStocks.put(book,newStock);
        }
        finally
        {
            writeLock.unlock();
        }
    }

    public int getStock(Book book)
    {
        readLock.lock();
        try
        {
            return bookStocks.getOrDefault(book,0);
        }
        finally
        {
            readLock.unlock();
        }
    }
}

class Order
{
    private List <Book> Books;
    private String UserID;

    public Order(List <Book> Books,String UserID)
    {
        this.Books = Books;
        this.UserID = UserID;
    }
}

class Review
{
    private String UserID;
    private Book book;
    private String ReviewText;
    private int Rating;

    public Review(String UserID,Book book,String ReviewText,int Rating)
    {
        this.UserID = UserID;
        this.book = book;
        this.ReviewText = ReviewText;
        this.Rating = Rating;
    }

    public String getUserID()
    {
        return UserID;
    }

    public Book getBook()
    {
        return book;
    }

    public String getReviewText()
    {
        return ReviewText;
    }

    public int getRating()
    {
        return Rating;
    }
}

class Bookstore
{
    private List <User> Users;
    private List <Book> bookCatalog;
    private Map <String,ShoppingCart> shoppingCarts;
    private Inventory inventory;
    private List <Order> orders;
    private List <Review> reviews;

    public Bookstore()
    {
        this.Users = new ArrayList <> ();
        this.bookCatalog = new ArrayList <> ();
        this.shoppingCarts = new HashMap <> ();
        this.inventory = new Inventory();
        this.orders = new ArrayList <> ();
        this.reviews = new ArrayList <> ();
    }

    public void registerUser(String UserID,String Username,String Password)
    {
        User user = new User(UserID,Username,Password);
        Users.add(user);
    }

    public User login(String UserID,String Password)
    {
        for (User user : Users)
        {
            if (user.getUserID().equals(UserID) && user.getPassword().equals(Password))
            {
                return user;
            }
        }
        return null;
    }

    public void addToCatalog(Book book)
    {
        bookCatalog.add(book);
    }

    public List <Book> searchBooksByTitle(String Title)
    {
        List <Book> Results = new ArrayList <> ();
        for (Book book : bookCatalog)
        {
            if (book.getTitle().equalsIgnoreCase(Title))
            {
                Results.add(book);
            }
        }
        return Results;
    }

    public List <Book> searchBooksByAuthor(String Author)
    {
        List <Book> Results = new ArrayList <> ();
        for (Book book : bookCatalog)
        {
            if (book.getAuthor().equalsIgnoreCase(Author))
            {
                Results.add(book);
            }
        }
        return Results;
    }

    public void addToCart(String UserID,Book book)
    {
        ShoppingCart Cart = shoppingCarts.getOrDefault(UserID,new ShoppingCart());
        Cart.addBook(book);
        shoppingCarts.put(UserID,Cart);
    }

    public void removeFromCart(String UserID,Book book)
    {
        ShoppingCart Cart = shoppingCarts.get(UserID);
        if (Cart != null)
        {
            Cart.removeBook(book);
        }
    }

    public List <Book> getCartBooks(String UserID)
    {
        ShoppingCart Cart = shoppingCarts.get(UserID);
        if (Cart != null)
        {
            return Cart.getBooks();
        }
        return Collections.emptyList();
    }

    public void updateStock(Book book,int newStock)
    {
        inventory.updateStock(book,newStock);
    }

    public int getStock(Book book)
    {
        return inventory.getStock(book);
    }

    public void processOrder(String UserID)
    {
        ShoppingCart Cart = shoppingCarts.get(UserID);
        if (Cart != null)
        {
            Order order = new Order(Cart.getBooks(),UserID);
            orders.add(order);
            shoppingCarts.remove(UserID);
        }
    }

    public void addReview(String UserID,Book book,String ReviewText,int Rating)
    {
        Review review = new Review(UserID,book,ReviewText,Rating);
        reviews.add(review);
    }

    public List <Review> getBookReviews(Book book)
    {
        List <Review> bookReviews = new ArrayList <> ();
        for (Review review : reviews)
        {
            if (review.getBook().equals(book))
            {
                bookReviews.add(review);
            }
        }
        return bookReviews;
    }

    public double getAverageRating(Book book)
    {
        List <Review> bookReviews = getBookReviews(book);
        if (bookReviews.isEmpty())
        {
            return 0.0;
        }

        double totalRating = 0.0;
        for (Review review : bookReviews)
        {
            totalRating += review.getRating();
        }

        return totalRating / bookReviews.size();
    }
}

public class OnlineBookStore
{
    public static void main(String[] args)
    {
        Bookstore bookstore = new Bookstore();

        bookstore.registerUser("User1","Krish","pwd1");
        bookstore.registerUser("User2","Rohit","pwd2");
        bookstore.registerUser("User3","Shreya","pwd3");

        Book book1 = new Book("The Count of Monte Cristo","Alexandre Dumas",23.79,10);
        Book book2 = new Book("The Three Musketeers","Alexandre Dumas",9.96,5);
        Book book3 = new Book("Treasure Island","RL Stevenson",28.97,10);
        Book book4 = new Book("Around The World In Eighty Days","Jules Verne",7.37,5);

        bookstore.addToCatalog(book1);
        bookstore.addToCatalog(book2);
        bookstore.addToCatalog(book3);
        bookstore.addToCatalog(book4);

        List <Book> booksByTitle = bookstore.searchBooksByTitle("Treasure Island");
        List <Book> booksByAuthor = bookstore.searchBooksByAuthor("Alexandre Dumas");

        System.out.println("Books by title 'Treasure Island': ");
        for (Book book : booksByTitle)
        {
            System.out.println("Title: " + book.getTitle());
            System.out.println("Author: " + book.getAuthor());
            System.out.println("Price: " + book.getPrice());
            System.out.println("Stock: " + book.getStock());
            System.out.println("-----------------------------");
        }

        System.out.println("Books by author 'Alexandre Dumas': ");

        for (Book book : booksByAuthor)
        {
            System.out.println("Title: " + book.getTitle());
            System.out.println("Author: " + book.getAuthor());
            System.out.println("Price: " + book.getPrice());
            System.out.println("Stock: " + book.getStock());
            System.out.println("-----------------------------");
        }
        bookstore.addToCart("Krish",book2);
        bookstore.addToCart("Shreya",book2);

        List <Book> user3CartBooks = bookstore.getCartBooks("Shreya");
        bookstore.updateStock(book2,5);
        int book2Stock = bookstore.getStock(book2);
        bookstore.processOrder("Krish");

        bookstore.addReview("Krish",book2,"Great book!",5);
        bookstore.addReview("Shreya",book2,"Average book.",3);

        List <Review> book2Reviews = bookstore.getBookReviews(book2);
        System.out.println("Reviews for book: " + book2.getTitle());
        for (Review review : book2Reviews)
        {
            System.out.println("User: " + review.getUserID());
            System.out.println("Rating: " + review.getRating());
            System.out.println("Review: " + review.getReviewText());
            System.out.println("-----------------------------");
        }

        double book2AverageRating = bookstore.getAverageRating(book2);
        System.out.println("Average rating for book: " + book2.getTitle() + ": " + book2AverageRating);
    }
}